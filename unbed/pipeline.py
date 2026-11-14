"""Remove background music and keep the voice.

process(source, job_dir, ...) -> dict with keys:
  chosen, alt, credit, metadata, quality_notes, choice, fidelity
  and optionally "small" if a compact 1080p copy was requested.

Method:
  ingest a URL (yt-dlp, 1080p cap, original audio) or a local video file
  demucs htdemucs_ft --two-stems vocals --segment 7, auto device (CUDA / MPS / CPU)
  soxr resample stems to the source rate, fidelity analysis
  full-removal mix + attenuated alt (default -25 dB); full removal unless the vocal stem is damaged
  two-pass linear loudnorm to -16 LUFS (TP raised if needed), AAC 256k, -c:v copy
  trim music-only tail > 5 s, credit.md in the original language
  delete work files unless keep_temp=True
"""
import os, re, sys, json, glob, shutil, subprocess, unicodedata, datetime, logging
import numpy as np
import soundfile as sf

# Use whatever interpreter is running this package (the venv's python).
PY = sys.executable
YTDLP = [PY, "-m", "yt_dlp"]
MAX_DURATION = 30 * 60
SMALL_LIMIT = 50 * 1024 * 1024
TARGET_I = -16.0
TAIL_MIN = 5.0
log = logging.getLogger("pipeline")


def _env_float(name, default):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        log.warning("Ignoring %s=%r; using %s", name, v, default)
        return default


def _env_int(name, default):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        log.warning("Ignoring %s=%r; using %s", name, v, default)
        return default


def _ytdlp_cookies(cookies=None):
    """Optional Netscape cookies file for login-required videos."""
    p = cookies or os.environ.get("UNBED_YTDLP_COOKIES") or os.environ.get("UNBED_COOKIES")
    return ["--cookies", p] if p else []


def _mps_available(torch):
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def select_device(prefer=None):
    """Pick a Demucs device: prefer / UNBED_DEVICE=cpu|cuda|mps, or auto (CUDA, else MPS, else CPU)."""
    import torch
    forced = (prefer or os.environ.get("UNBED_DEVICE") or "").strip().lower()
    if forced:
        if forced not in ("cpu", "cuda", "mps"):
            log.warning("Unknown UNBED_DEVICE=%r; auto-detecting instead", forced)
        else:
            if forced == "cuda" and not torch.cuda.is_available():
                log.warning("UNBED_DEVICE=cuda but CUDA is not available; using CPU")
                return "cpu"
            if forced == "mps" and not _mps_available(torch):
                log.warning("UNBED_DEVICE=mps but MPS is not available; using CPU")
                return "cpu"
            return forced
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def _demucs_cmd(audio, sep, device):
    # htdemucs_ft max segment is about 7.8; 7 is safe on CPU and GPU.
    segment = _env_int("UNBED_DEMUCS_SEGMENT", 7)
    cmd = [PY, "-m", "demucs", "-n", "htdemucs_ft", "--two-stems", "vocals",
           "--segment", str(segment), "--float32", "--clip-mode", "none",
           "-d", device]
    # -j runs several chunks at once. On a GPU that multiplies memory use.
    if device == "cpu":
        cmd += ["-j", str(_env_int("UNBED_DEMUCS_JOBS", 2))]
    cmd += ["-o", sep, audio]
    return cmd

LOGIN_PATTERNS = re.compile(
    r"(sign in to confirm|login required|private video|this video is private|"
    r"age[- ]restricted|confirm your age|members[- ]only|requires authentication|"
    r"use --cookies|not available without|you need to log in|NSFW tweet|"
    r"account.*(suspended|protected))", re.I)


class PipelineError(Exception):
    """Error with a short message for the user (self.pt) and a technical message (args)."""
    def __init__(self, msg, pt=None):
        super().__init__(msg)
        self.pt = pt or "Processing failed."


class LoginRequired(PipelineError):
    pass


class TooLong(PipelineError):
    pass


def run(cmd, timeout=None, capture=True):
    log.debug("run: %s", cmd[:8])
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True, errors="replace", timeout=timeout)
    except OSError as e:
        raise PipelineError("cannot run %s: %s" % (cmd[0], e),
                            pt="Could not run %s (%s)." % (cmd[0], e.strerror)) from e
    except subprocess.TimeoutExpired as e:
        raise PipelineError("timeout %s" % cmd[0],
                            pt="%s took too long and was stopped." % cmd[0]) from e
    return p


def _ytdlp_fail(p, what):
    err = (p.stderr or "")[-3000:]
    if LOGIN_PATTERNS.search(err):
        raise LoginRequired("%s needs login/cookies: %s" % (what, err.strip()[-400:]),
                            pt="This link requires login/cookies (private, age-restricted, or members-only). "
                               "Pass --cookies with a Netscape cookies file and try again.")
    raise PipelineError("%s failed: %s" % (what, err.strip()[-600:]),
                        pt="Could not download this video (the link may be broken or unavailable).")


# ---------------------------------------------------------------- download
def probe(url, cookies=None):
    """yt-dlp --dump-json (no download). Raises LoginRequired / TooLong."""
    p = run(YTDLP + _ytdlp_cookies(cookies) + ["--dump-json", "--no-playlist", "--playlist-items", "1", "--no-warnings", "--", url], timeout=180)
    if p.returncode != 0 or not (p.stdout or "").strip():
        _ytdlp_fail(p, "probe")
    raw = p.stdout.strip().splitlines()[0]
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PipelineError("probe json: %s" % raw[:200],
                            pt="Could not read the video information.") from e
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise PipelineError("live", pt="This link is a live or upcoming stream; only already-published videos are supported.")
    dur = info.get("duration")
    if dur and dur > MAX_DURATION:
        raise TooLong("too long %s" % dur, pt="This video is %s long; the limit is 30 minutes." % fmt_t(dur))
    return info


def video_key(info):
    ext = (info.get("extractor_key") or info.get("extractor") or "x").lower()
    vid = re.sub(r"[^A-Za-z0-9_-]", "_", str(info.get("id")))
    return vid if ext == "youtube" else "%s_%s" % (ext, vid)


def is_url(source):
    return bool(re.match(r"^https?://", str(source).strip(), re.I))


def local_info(path, original_path=None):
    """Build a yt-dlp-like info dict from a local file so credit.md still works."""
    pr = ffprobe(path)
    vs = next((st for st in pr.get("streams") or [] if st.get("codec_type") == "video"), {})
    dur = float((pr.get("format") or {}).get("duration") or 0)
    src = original_path or path
    title = Path_stem(src)
    ident = re.sub(r"[^A-Za-z0-9_-]", "_", title)[:80] or "local"
    return {
        "id": ident,
        "title": title,
        "extractor_key": "local",
        "duration": dur,
        "width": vs.get("width"),
        "height": vs.get("height"),
        "webpage_url": os.path.abspath(src),
        "channel": None,
        "uploader": None,
        "description": "",
        "upload_date": None,
        "language": None,
        "vcodec": vs.get("codec_name"),
    }


def Path_stem(src):
    return os.path.splitext(os.path.basename(src))[0]


def ingest(source, work, cookies=None):
    """Return (media_path, info) from a URL or a local video file."""
    os.makedirs(work, exist_ok=True)
    if not is_url(source) and os.path.isfile(source):
        ext = os.path.splitext(source)[1] or ".mp4"
        dest = os.path.join(work, "src" + ext)
        shutil.copy2(source, dest)
        info = local_info(dest, original_path=source)
        with open(os.path.join(work, "src.info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        return dest, info
    if not is_url(source):
        raise PipelineError("not found", pt="No such file or URL: %s" % source)
    return download(source, work, cookies=cookies)


def download(url, work, cookies=None):
    os.makedirs(work, exist_ok=True)
    # Original audio track first (YouTube marks it "original" in format_note / highest language_preference);
    # auto-dubbed tracks are never chosen unless nothing else exists.
    # Cap at 1080p (landscape height<=1080 / portrait width<=1080) so mux can keep -c:v copy.
    _h1080 = "[height<=1080][width<=1920]"
    _v1080 = "[width<=1080][height<=1920]"
    _ba_orig = "ba[format_note*=original]"
    _ba_nodub = "ba[format_note!*=dubbed][format_note!*=Dubbed]"
    fmt = "/".join([
        f"bv*{_h1080}+{_ba_orig}", f"bv*{_v1080}+{_ba_orig}",
        f"bv*{_h1080}+{_ba_nodub}", f"bv*{_v1080}+{_ba_nodub}",
        f"bv*{_h1080}+ba", f"bv*{_v1080}+ba",
        f"b{_h1080}", f"b{_v1080}",
    ])
    cmd = YTDLP + _ytdlp_cookies(cookies) + ["--no-playlist", "--playlist-items", "1", "--no-warnings", "--no-progress",
                   "-f", fmt, "-S", "lang,res,fps,br", "--merge-output-format", "mp4", "--write-info-json",
                   "-o", os.path.join(work, "src.%(ext)s"), "--", url]
    p = run(cmd, timeout=3600)
    if p.returncode != 0:
        _ytdlp_fail(p, "download")
    vids = [f for f in glob.glob(os.path.join(work, "src.*")) if not f.endswith(".json") and not f.endswith(".part")]
    if not vids:
        raise PipelineError("no output file", pt="The download did not produce a video file.")
    vids.sort(key=lambda p: (0 if p.endswith(".mp4") else 1, p))
    return vids[0], _load_info_json(work)


def _load_info_json(work):
    direct = os.path.join(work, "src.info.json")
    if os.path.isfile(direct):
        path = direct
    else:
        matches = sorted(glob.glob(os.path.join(work, "*.info.json")))
        if len(matches) != 1:
            raise PipelineError("no info json", pt="The download did not include video information.")
        path = matches[0]
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    except json.JSONDecodeError as e:
        raise PipelineError("info json: %s" % e, pt="The download did not include video information.") from e
    if not isinstance(info, dict):
        raise PipelineError("info json", pt="The download did not include video information.")
    # process() always copies work/src.info.json into the job folder.
    if os.path.abspath(path) != os.path.abspath(direct):
        with open(direct, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
    return info


# ---------------------------------------------------------------- audio helpers
def ffprobe(path):
    p = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    if p.returncode != 0 or not (p.stdout or "").strip():
        raise PipelineError("ffprobe failed: %s" % ((p.stderr or "").strip()[-400:]),
                            pt="Could not read this media file.")
    try:
        info = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise PipelineError("ffprobe json", pt="Could not read this media file.") from e
    if not isinstance(info, dict):
        raise PipelineError("ffprobe json", pt="Could not read this media file.")
    return info


def fmt_t(s):
    s = int(round(s))
    return "%d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60) if s >= 3600 else "%d:%02d" % (s // 60, s % 60)


def ff(cmd, what):
    p = run(["ffmpeg", "-hide_banner", "-nostdin", "-y"] + cmd)
    if p.returncode != 0:
        raise PipelineError("%s: %s" % (what, p.stderr[-800:]), pt="ffmpeg failed (%s)." % what)
    return p


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def frame_rms(x, sr, hop=0.1):
    n = int(sr * hop)
    m = x.mean(axis=1) if x.ndim == 2 else x
    k = len(m) // n
    if k == 0:
        return np.array([0.0])
    return np.sqrt((m[:k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-20)


def centroid(x, sr):
    m = x.mean(axis=1) if x.ndim == 2 else x
    n = 4096
    k = len(m) // n
    if k == 0:
        return 0.0
    seg = m[:k * n].reshape(k, n) * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    return float((f * mag).sum() / max(mag.sum(), 1e-12))


def fidelity(orig, voc, rest, sr):
    """Compare stems with the original. Returns metrics + damaged flag + tail info."""
    L = min(len(orig), len(voc), len(rest))
    orig, voc, rest = orig[:L], voc[:L], rest[:L]
    recon_err = orig - (voc + rest)
    snr = db(np.sqrt((orig ** 2).mean()) / max(np.sqrt((recon_err ** 2).mean()), 1e-12))
    rv, ro, rm = frame_rms(voc, sr), frame_rms(orig, sr), frame_rms(rest, sr)
    vmax = rv.max() if len(rv) else 0
    voiced = rv > max(vmax * 10 ** (-35 / 20), 10 ** (-50 / 20))
    voiced_frac = float(voiced.mean()) if len(voiced) else 0.0
    # level/tone match in frames where the voice dominates (music < voice - 15 dB)
    dom = voiced & (rm < rv * 10 ** (-15 / 20))
    if dom.sum() >= 10:
        level_diff = db(rv[dom].mean()) - db(ro[dom].mean())
    else:
        level_diff = db(rv[voiced].mean() if voiced.any() else 1e-12) - db(ro[voiced].mean() if voiced.any() else 1e-12)
    hop = int(sr * 0.1)
    idx = np.repeat(voiced, hop)[:L] if voiced.any() else np.zeros(L, bool)
    if idx.sum() > sr:
        cent_v, cent_o = centroid(voc[:len(idx)][idx], sr), centroid(orig[:len(idx)][idx], sr)
    else:
        cent_v = cent_o = 0.0
    clip_frac = float((np.abs(voc) >= 0.999).mean())
    nan = bool(~np.isfinite(voc).all())
    music_db = db(np.sqrt((rest ** 2).mean())) - db(np.sqrt((voc ** 2).mean()) + 1e-12)
    # music-only tail
    total = L / sr
    last_voice = (np.nonzero(voiced)[0][-1] + 1) * 0.1 if voiced.any() else 0.0
    tail = total - last_voice
    reasons = []
    if nan:
        reasons.append("NaN in the vocal stem")
    if clip_frac > 1e-3:
        reasons.append("clipping in the vocal stem (%.2f%%)" % (clip_frac * 100))
    if snr < 20:
        reasons.append("weak reconstruction (%.1f dB)" % snr)
    if voiced_frac < 0.05:
        reasons.append("almost no voice detected (%.0f%%)" % (voiced_frac * 100))
    if abs(level_diff) > 3:
        reasons.append("voice level differs from the original by %.1f dB" % level_diff)
    return {
        "reconstruction_snr_db": round(snr, 1), "voice_level_diff_db": round(level_diff, 2),
        "centroid_voice_hz": round(cent_v), "centroid_orig_hz": round(cent_o),
        "voiced_fraction": round(voiced_frac, 3), "vocal_clip_fraction": clip_frac,
        "music_vs_voice_db": round(music_db, 1), "duration": round(total, 2),
        "last_voice_s": round(last_voice, 2), "music_only_tail_s": round(tail, 2),
        "damaged": bool(reasons), "damage_reasons": reasons,
    }


def _loudnorm_stats(text, required):
    """Return the last JSON object in ffmpeg's stderr that contains `required`."""
    decoder = json.JSONDecoder()
    start = (text or "").rfind("{")
    while start >= 0:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.rfind("{", 0, start)
            continue
        if isinstance(obj, dict) and required in obj:
            return obj
        start = text.rfind("{", 0, start)
    raise PipelineError("loudnorm stats", pt="ffmpeg did not report loudness stats.")


def _loudnorm_floats(stats, keys):
    try:
        return tuple(float(stats[key]) for key in keys)
    except (KeyError, TypeError, ValueError):
        raise PipelineError("loudnorm stats", pt="ffmpeg did not report loudness stats.") from None


def loudnorm_two_pass(src_wav, dst_wav):
    p = ff(["-i", src_wav, "-af", "loudnorm=I=%s:TP=-1.5:LRA=11:print_format=json" % TARGET_I, "-f", "null", "-"],
           "loudnorm pass 1")
    m = _loudnorm_stats(p.stderr, "input_i")
    ii, itp, ilra, ith = _loudnorm_floats(m, ("input_i", "input_tp", "input_lra", "input_thresh"))
    if not np.isfinite(ii) or ii < -70:
        raise PipelineError("silent audio", pt="The processed audio was silent.")
    target = TARGET_I
    tp = -1.5
    gain = target - ii
    if itp + gain > tp:            # linear would exceed TP -> raise TP
        tp = min(round(itp + gain + 0.1, 1), 0.0)
    if itp + gain > tp:            # still over 0 dBTP -> lower target to stay linear
        target = round(tp - 0.1 - itp + ii, 1)
    lra = min(max(11.0, round(ilra + 1, 1)), 50.0)
    # The wav is already at the source rate. Resampling again here colors the voice
    # and fails on ffmpeg builds that have no soxr resampler.
    af = ("loudnorm=I=%s:TP=%s:LRA=%s:measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:"
          "offset=%s:linear=true:print_format=json" %
          (target, tp, lra, ii, itp, ilra, ith, m.get("target_offset", 0.0)))
    p2 = ff(["-i", src_wav, "-af", af, "-c:a", "pcm_f32le", dst_wav], "loudnorm pass 2")
    m2 = _loudnorm_stats(p2.stderr, "output_i")
    out_i, out_tp = _loudnorm_floats(m2, ("output_i", "output_tp"))
    return {"input_i": ii, "input_tp": itp, "input_lra": ilra, "target_i": target, "target_tp": tp,
            "output_i": out_i, "output_tp": out_tp,
            "type": m2.get("normalization_type")}


def mux(video_src, audio_wav, out, trim_to=None):
    cmd = ["-i", video_src, "-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest"]
    if trim_to:
        cmd += ["-t", "%.3f" % trim_to]
    ff(cmd + [out], "mux")


def compact_copy(src, out, duration):
    """libx264 1080p copy under 50 MB. Returns None when src is already small enough."""
    if os.path.getsize(src) <= SMALL_LIMIT:
        return None
    budget = 46.5 * 1024 * 1024 * 8
    for factor in (1.0, 0.85, 0.7, 0.55):
        vbr = int(min(8_000_000, (budget / max(duration, 1) - 128_000) * factor))
        if vbr < 150_000:
            raise PipelineError("too long for compact copy", pt="The video is too long to fit in a 50 MB compact copy.")
        ff(["-i", src, "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=-2:'min(1080,ih)'",
            "-c:v", "libx264", "-preset", "veryfast", "-b:v", str(vbr), "-maxrate", str(int(vbr * 1.3)),
            "-bufsize", str(int(vbr * 2)), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", out], "compact copy")
        if os.path.getsize(out) <= SMALL_LIMIT:
            return out
    raise PipelineError("compact copy still too big", pt="Could not shrink the video under 50 MB.")


# ---------------------------------------------------------------- credit.md
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def hashtag(s):
    w = re.sub(r"[^A-Za-z0-9 ]", " ", strip_accents(s)).split()
    return "".join(x[:1].upper() + x[1:] for x in w)[:40]


def safe_name(s):
    s = re.sub(r"[^\w\s.,()&'-]", "", strip_accents(s)).strip()
    return re.sub(r"\s+", " ", s)[:90] or "video"


L10N = {
    "pt": dict(head="Créditos", titles="Sugestões de título", caps="Legendas", meta="Metadados", warn="Aviso",
               nomusic="sem música", speech="só a fala, sem trilha", credit="crédito", original="Original",
               channel="canal", yt_line="versão com a música de fundo removida para uso informativo e educativo.",
               orig_video="Vídeo original", cr="Crédito", notaltered="A fala não foi alterada; apenas a trilha sonora foi removida/atenuada. Todos os direitos pertencem aos seus respectivos donos.",
               ig_line="Versão sem música de fundo, para uso informativo e educativo.", ig_rights="Sem reivindicação de autoria; direitos dos respectivos donos.",
               x_line="versão sem música de fundo (uso informativo/educativo).", on="no",
               m_url="URL original", m_chan="Canal/autor do envio", m_title="Título original", m_desc="Descrição original",
               m_date="Data de publicação", m_dur="Duração", m_lang="Idioma", m_res="Resolução", m_proc="Data de processamento",
               m_trim="Corte", trimmed="trecho final só com música removido (%s–%s)",
               lang_note="faixa de áudio original", res_note="vídeo original sem recodificação",
               disclaimer="A música de fundo deste vídeo foi removida (versão \"sem musica\") ou atenuada (versão \"musica atenuada\") apenas para uso informativo e educativo. A fala original foi preservada sem alterações de tom, velocidade ou timbre. Não reivindicamos a propriedade do conteúdo; todos os direitos pertencem ao criador original e aos respectivos titulares.",
               datefmt="%d/%m/%Y"),
    "es": dict(head="Créditos", titles="Sugerencias de título", caps="Textos", meta="Metadatos", warn="Aviso",
               nomusic="sin música", speech="solo la voz, sin música", credit="crédito", original="Original",
               channel="canal", yt_line="versión con la música de fondo eliminada para uso informativo y educativo.",
               orig_video="Video original", cr="Crédito", notaltered="La voz no fue alterada; solo se eliminó/atenuó la música. Todos los derechos pertenecen a sus respectivos dueños.",
               ig_line="Versión sin música de fondo, para uso informativo y educativo.", ig_rights="Sin reclamar autoría; derechos de sus respectivos dueños.",
               x_line="versión sin música de fondo (uso informativo/educativo).", on="en",
               m_url="URL original", m_chan="Canal/autor", m_title="Título original", m_desc="Descripción original",
               m_date="Fecha de publicación", m_dur="Duración", m_lang="Idioma", m_res="Resolución", m_proc="Fecha de procesamiento",
               m_trim="Recorte", trimmed="se quitó el tramo final solo con música (%s–%s)",
               lang_note="pista de audio original", res_note="video original sin recodificar",
               disclaimer="La música de fondo de este video fue eliminada (versión \"sem musica\") o atenuada (versión \"musica atenuada\") solo para uso informativo y educativo. La voz original se conservó sin cambios de tono, velocidad ni timbre. No reclamamos la propiedad del contenido; todos los derechos pertenecen al creador original y a sus titulares.",
               datefmt="%d/%m/%Y"),
    "en": dict(head="Credits", titles="Title suggestions", caps="Captions", meta="Metadata", warn="Notice",
               nomusic="no music", speech="speech only, no soundtrack", credit="credit", original="Original",
               channel="channel", yt_line="version with the background music removed, for informational and educational use.",
               orig_video="Original video", cr="Credit", notaltered="The speech was not altered; only the soundtrack was removed/attenuated. All rights belong to their respective owners.",
               ig_line="Version without background music, for informational and educational use.", ig_rights="No claim of authorship; rights belong to their respective owners.",
               x_line="version without background music (informational/educational use).", on="on",
               m_url="Original URL", m_chan="Channel/uploader", m_title="Original title", m_desc="Original description",
               m_date="Published", m_dur="Duration", m_lang="Language", m_res="Resolution", m_proc="Processed",
               m_trim="Trim", trimmed="music-only ending removed (%s–%s)",
               lang_note="original audio track", res_note="original video, not re-encoded",
               disclaimer="The background music in this video was removed (\"sem musica\" version) or attenuated (\"musica atenuada\" version) for informational and educational use only. The original speech was preserved with no change to pitch, speed or timbre. We claim no ownership of the content; all rights belong to the original creator and rights holders.",
               datefmt="%Y-%m-%d"),
}
LANG_NAMES = {"pt": "português", "es": "español", "en": "English"}


def detect_lang(info):
    for k in ("language",):
        v = info.get(k)
        if v:
            return v
    # YouTube: the original audio format carries the language
    for f in info.get("formats") or []:
        if "original" in str(f.get("format_note", "")).lower() and f.get("language"):
            return f["language"]
    return None


def write_credit(path, info, url, fid, trim, lang_full):
    lang = (lang_full or "en").split("-")[0].lower()
    T = L10N.get(lang, L10N["en"])
    raw_title = (info.get("title") or "").strip()
    title_tags = [hashtag(t) for t in re.findall(r"#(\w+)", raw_title)]
    title = re.sub(r"\s*#\w+", "", raw_title).strip() or raw_title
    who = info.get("channel") or info.get("uploader") or info.get("uploader_id") or "?"
    handle = info.get("uploader_id") or ""
    if handle and not handle.startswith("@") and info.get("extractor_key", "").lower() in ("twitter", "instagram"):
        handle = "@" + handle
    chan_url = info.get("channel_url") or info.get("uploader_url") or ""
    site = {"youtube": "YouTube", "instagram": "Instagram", "twitter": "X"}.get(
        (info.get("extractor_key") or "").lower(), info.get("extractor_key") or "")
    page = info.get("webpage_url") or url
    who_h = "%s (%s)" % (who, handle) if handle and handle != who else who
    if len(title) > 70 or not title:
        short = (title[:67] + "…") if title else ("%s — %s" % (who, site))
    else:
        short = title
    t1 = "%s (%s) | %s" % (short, T["nomusic"], who)
    t2 = "\"%s\" — %s | %s: %s %s" % (short, T["speech"], T["original"], T["channel"], who)
    t3 = "%s (%s) — %s: %s" % (short, T["nomusic"], T["credit"], handle or who)
    tags = []
    first = re.split(r"\s[-–—|:]\s|:", title)[0]
    for t in [hashtag(who)] + title_tags + ([hashtag(first)] if 3 <= len(first) <= 28 else []):
        if t and t not in tags:
            tags.append(t)
    tags_s = " ".join("#" + t for t in tags)
    pub = "?"
    d = info.get("upload_date")
    if d:
        try:
            pub = datetime.datetime.strptime(str(d), "%Y%m%d").strftime(T["datefmt"])
        except ValueError:
            pub = "?"
    desc = (info.get("description") or "").strip().replace("\n", " ")
    desc = (desc[:300] + "…") if len(desc) > 300 else desc
    dur = info.get("duration") or fid["duration"]
    res = "%s×%s" % (info.get("width") or "?", info.get("height") or "?")
    lines = [
        "# %s — \"%s\"" % (T["head"], title or page), "",
        "## %s" % T["titles"], "1. " + t1, "2. " + t2, "3. " + t3, "",
        "## %s" % T["caps"], "", "### YouTube",
        "%s (%s) — %s" % (short, who, T["yt_line"][:1].upper() + T["yt_line"][1:]), "",
        "🎥 %s: %s" % (T["orig_video"], page),
        "📺 %s: %s %s%s" % (T["cr"], T["channel"], who_h, (" — " + chan_url) if chan_url else ""), "",
        T["notaltered"], "", tags_s, "",
        "### Instagram", "%s — %s" % (short, who), T["ig_line"], "",
        "🎥 %s: %s" % (T["original"], page), "📺 %s: %s (%s)" % (T["cr"], who_h, site), "",
        T["ig_rights"], tags_s, "",
        "### X", "%s — %s, %s" % (short, who, T["x_line"]), "%s: %s" % (T["original"], page),
        "%s: %s %s %s" % (T["cr"], who_h, T["on"], site), tags_s.split(" ")[0] if tags_s else "", "",
        "## %s" % T["meta"],
        "- %s: %s" % (T["m_url"], page),
        "- %s: %s%s" % (T["m_chan"], who_h, (" — " + chan_url) if chan_url else ""),
        "- %s: %s" % (T["m_title"], raw_title),
        "- %s: \"%s\"" % (T["m_desc"], desc),
        "- %s: %s" % (T["m_date"], pub),
        "- %s: %s (%d s)" % (T["m_dur"], fmt_t(dur), round(dur)),
        "- %s: %s (%s) — %s" % (T["m_lang"], LANG_NAMES.get(lang, lang), lang_full or "?", T["lang_note"]),
        "- %s: %s, %s" % (T["m_res"], res, T["res_note"]),
    ]
    if trim:
        lines.append("- %s: %s" % (T["m_trim"], T["trimmed"] % (fmt_t(trim), fmt_t(fid["duration"]))))
    lines += ["- %s: %s" % (T["m_proc"], datetime.date.today().strftime(T["datefmt"])), "",
              "## %s" % T["warn"], T["disclaimer"], ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"title": t1, "lang": lang}


# ---------------------------------------------------------------- main
def process(source, job_dir, keep_temp=False, cookies=None, device=None, alt_db=None, small=False):
    """Ingest a URL or local file, write stems/mixes and credit.md under job_dir."""
    os.makedirs(job_dir, exist_ok=True)
    work = os.path.join(job_dir, "work")
    final = os.path.join(job_dir, "final")
    os.makedirs(final, exist_ok=True)
    notes = []
    att = _env_float("UNBED_ALT_DB", -25.0) if alt_db is None else float(alt_db)
    try:
        src, info = ingest(source, work, cookies=cookies)
        shutil.copy(os.path.join(work, "src.info.json"), os.path.join(job_dir, "%s.info.json" % video_key(info)))
        pr = ffprobe(src)
        astreams = [st for st in (pr.get("streams") or []) if st.get("codec_type") == "audio"]
        if not astreams:
            raise PipelineError("no audio", pt="This video has no audio.")
        try:
            duration = float((pr.get("format") or {})["duration"])
        except (KeyError, TypeError, ValueError):
            raise PipelineError("no duration", pt="Could not read the duration of this video.") from None
        if duration > MAX_DURATION + 5:
            raise TooLong("too long", pt="This video is %s long; the limit is 30 minutes." % fmt_t(duration))
        audio = os.path.join(work, "audio.wav")
        ff(["-i", src, "-map", "0:a:0", "-ac", "2", "-c:a", "pcm_f32le", audio], "extract audio")
        try:
            sr = int(sf.info(audio).samplerate)
        except (OSError, sf.SoundFileError, RuntimeError) as e:
            raise PipelineError("wav rate: %s" % e, pt="Could not read the extracted audio.") from e
        sep = os.path.join(work, "sep")
        used = select_device(prefer=device)
        tag = "override" if device or (os.environ.get("UNBED_DEVICE") or "").strip() else "auto"
        extra = ""
        try:
            import torch
            if used == "cuda" and torch.cuda.is_available():
                extra = " (%s)" % torch.cuda.get_device_name(0)
        except Exception:
            pass
        log.info("Demucs device: %s%s [%s]", used, extra, tag)
        p = run(_demucs_cmd(audio, sep, used), timeout=4 * 3600)
        if p.returncode != 0 and used == "mps":
            log.warning("Demucs failed on MPS; falling back to CPU")
            used = "cpu"
            p = run(_demucs_cmd(audio, sep, used), timeout=4 * 3600)
        with open(os.path.join(job_dir, "demucs.log"), "w") as f:
            f.write(("device=%s\n" % used) + (p.stderr or "")[-20000:])
        if p.returncode != 0:
            raise PipelineError("demucs failed", pt="Voice separation (Demucs) failed.")
        stems = os.path.join(sep, "htdemucs_ft", "audio")
        rs = {}
        for name in ("vocals", "no_vocals"):
            out = os.path.join(work, name + "_rs.wav")
            ff(["-i", os.path.join(stems, name + ".wav"), "-af", "aresample=%d:resampler=soxr" % sr,
                "-c:a", "pcm_f32le", out], "resample")
            rs[name] = out
        orig, _ = sf.read(audio, dtype="float32", always_2d=True)
        voc, _ = sf.read(rs["vocals"], dtype="float32", always_2d=True)
        rest, _ = sf.read(rs["no_vocals"], dtype="float32", always_2d=True)
        L = min(len(orig), len(voc), len(rest))
        fid = fidelity(orig[:L], voc[:L], rest[:L], sr)
        del orig
        sf.write(os.path.join(work, "full.wav"), voc[:L], sr, subtype="FLOAT")
        alt_mix = voc[:L] + rest[:L] * np.float32(10 ** (att / 20))
        del voc, rest
        sf.write(os.path.join(work, "alt.wav"), alt_mix, sr, subtype="FLOAT")
        del alt_mix
        choice = "alt" if fid["damaged"] else "full"
        if fid["damaged"]:
            notes.append("Vocal stem looks damaged (%s) — used the attenuated-music mix (%.0f dB)."
                         % ("; ".join(fid["damage_reasons"]), att))
        trim = None
        if fid["music_only_tail_s"] > TAIL_MIN and fid["last_voice_s"] > 1:
            trim = min(fid["last_voice_s"] + 1.0, duration)
            notes.append("Music-only ending (%s–%s) trimmed from the no-music version."
                         % (fmt_t(fid["last_voice_s"]), fmt_t(fid["duration"])))
        ln = {}
        for k in ("full", "alt"):
            ln[k] = loudnorm_two_pass(os.path.join(work, k + ".wav"), os.path.join(work, k + "_ln.wav"))
            if ln[k]["type"] != "linear":
                notes.append("loudnorm (%s) came out as %s, not linear." % (k, ln[k]["type"]))
            if ln[k]["target_i"] != TARGET_I:
                notes.append("Loudness (%s) set to %.1f LUFS to stay linear at the peaks." % (k, ln[k]["target_i"]))
        base = safe_name(re.sub(r"\s*#\w+", "", info.get("title") or "") or video_key(info))
        full_mp4 = os.path.join(final, base + " - sem musica.mp4")
        alt_mp4 = os.path.join(final, base + " - musica atenuada.mp4")
        mux(src, os.path.join(work, "full_ln.wav"), full_mp4, trim_to=trim)
        mux(src, os.path.join(work, "alt_ln.wav"), alt_mp4)
        chosen = full_mp4 if choice == "full" else alt_mp4
        small_path = None
        if small:
            chosen_dur = float(ffprobe(chosen)["format"]["duration"])
            compact = os.path.join(final, base + " (1080p small).mp4")
            small_path = compact_copy(chosen, compact, chosen_dur)
        lang_full = detect_lang(info)
        if not lang_full:
            notes.append("Source language was not provided; credits written in English.")
        credit = os.path.join(final, "credit.md")
        cr = write_credit(credit, info, info.get("webpage_url") or source, fid, trim, lang_full)
        notes.insert(0, "Voice vs original: %+.2f dB level, brightness %d vs %d Hz; music was %.1f dB relative to the voice."
                     % (fid["voice_level_diff_db"], fid["centroid_voice_hz"], fid["centroid_orig_hz"],
                        fid["music_vs_voice_db"]))
        meta = {
            "source": source, "webpage_url": info.get("webpage_url"), "key": video_key(info), "id": info.get("id"),
            "extractor": info.get("extractor_key"), "title": info.get("title"),
            "uploader": info.get("channel") or info.get("uploader"), "duration": duration,
            "language": lang_full, "lang": cr["lang"], "width": info.get("width"), "height": info.get("height"),
            "vcodec": info.get("vcodec"), "audio_format": info.get("format_id"), "sample_rate": sr,
            "loudnorm": ln, "trim_to": trim, "device": used, "attenuation_db": att,
        }
        result = {"chosen": chosen, "alt": alt_mp4, "credit": credit, "metadata": meta,
                  "quality_notes": notes, "choice": choice, "fidelity": fid}
        if small_path:
            result["small"] = small_path
        with open(os.path.join(job_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        return result
    finally:
        if not keep_temp:
            shutil.rmtree(work, ignore_errors=True)
