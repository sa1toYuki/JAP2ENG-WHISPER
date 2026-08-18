import argparse
import sys
from pathlib import Path

# Particulas gramaticais cuja pronuncia real diverge da leitura padrao de
# dicionario. So aplicamos a excecao quando o token e classificado como
# particula (joshi) pelo Sudachi, para nao afetar o uso do mesmo kana
# dentro de outras palavras (ex: "haha", mae, continua "ha" normalmente).
PARTICLE_OVERRIDES = {
    "は": "wa",
    "へ": "e",
    "を": "o",
}

# Classes gramaticais do Sudachi tratadas como pontuacao/simbolo: coladas
# na palavra anterior, sem espaco e sem tentar romanizar.
SYMBOL_POS = {"補助記号", "記号"}

# Categorias que representam conjugacao/sufixo verbal (nao uma palavra nova):
# devem colar na palavra anterior sem espaco, para nao fragmentar verbos como
# "kaeshite" (返して) em "kaeshi te". Formato: (pos[0], pos[1]) ou (pos[0], None)
# quando qualquer subcategoria conta.
ATTACH_NO_SPACE_POS = {
    ("助詞", "接続助詞"),  # ex: て/で/し/ながら - conecta verbo a outro trecho
    ("助動詞", None),      # ex: た/ます/ない/だろう - conjugacao/auxiliar verbal
}


def _attaches_without_space(pos_tuple) -> bool:
    major, minor = pos_tuple[0], pos_tuple[1]
    return (major, minor) in ATTACH_NO_SPACE_POS or (major, None) in ATTACH_NO_SPACE_POS


def format_ass_time(seconds: float) -> str:
    """Converte segundos (float) para o formato de tempo do ASS: H:MM:SS.CC"""
    if seconds < 0:
        seconds = 0
    total_cs = round(seconds * 100)  # centissegundos
    h, rem = divmod(total_cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


ASS_HEADER = """[Script Info]
Title: Transcricao JP -> Romaji (auto-gerado, requer QC humano)
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def transcribe(audio_path: str, model_size: str, device: str, language: str = "ja"):
    import whisper

    print(f"[+] Carregando modelo Whisper '{model_size}' em '{device}'...", file=sys.stderr)
    model = whisper.load_model(model_size, device=device)

    print(f"[+] Transcrevendo audio: {audio_path}", file=sys.stderr)
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        verbose=False,
        word_timestamps=False,
    )
    return result["segments"]


def build_sudachi_tokenizer():
    from sudachipy import Dictionary

    print("[+] Carregando dicionario do SudachiPy...", file=sys.stderr)
    return Dictionary().create()


def get_split_mode(mode_str: str):
    from sudachipy import SplitMode

    return {"A": SplitMode.A, "B": SplitMode.B, "C": SplitMode.C}[mode_str]


def katakana_to_romaji(text: str, kks) -> str:
    """Converte katakana/kana -> romaji Hepburn via pykakasi."""
    result = kks.convert(text)
    return "".join(item["hepburn"] for item in result)


def romanize_with_tokens(text: str, sudachi_tokenizer, split_mode, kks) -> str:
    """
    Tokeniza o texto japones, resolve a leitura de cada palavra pelo contexto
    (evita erro de kanji ambiguo), aplica correcao de particulas gramaticais,
    e junta tudo com espacos entre palavras (pontuacao colada, sem espaco).
    """
    tokens = sudachi_tokenizer.tokenize(text, split_mode)
    words = []

    for tok in tokens:
        surface = tok.surface()
        if not surface.strip():
            continue

        pos = tok.part_of_speech()[0]

        # Pontuacao/simbolo: gruda na palavra anterior, sem romanizar.
        if pos in SYMBOL_POS:
            if words:
                words[-1] += surface
            else:
                words.append(surface)
            continue

        # Particula gramatical com pronuncia especial (ha/e/wo).
        if pos == "助詞" and surface in PARTICLE_OVERRIDES:
            words.append(PARTICLE_OVERRIDES[surface])
            continue

        reading = tok.reading_form()  # leitura em katakana, resolvida por contexto
        if reading and reading != "*":
            romaji = katakana_to_romaji(reading, kks)
        else:
            # Sem entrada no dicionario (nomes raros, ruido, texto nao-JP que
            # o Whisper "alucinou"). Mantem o texto original para o QC ver
            # facilmente que aquele trecho precisa de atencao manual.
            romaji = surface

        if not romaji:
            continue

        # Sufixo de conjugacao/auxiliar verbal: cola na palavra anterior
        # sem espaco (ex: "kaeshi" + "te" -> "kaeshite").
        if _attaches_without_space(tok.part_of_speech()) and words:
            words[-1] += romaji
        else:
            words.append(romaji)

    return " ".join(words)


def sanitize_for_ass(text: str) -> str:
    """Evita quebra de linha crua dentro do evento do ASS."""
    return text.replace("\n", " ").replace("\r", " ").strip()


def build_ass(segments, output_path: str, sudachi_mode: str):
    import pykakasi

    kks = pykakasi.kakasi()
    sudachi_tokenizer = build_sudachi_tokenizer()
    split_mode = get_split_mode(sudachi_mode)

    lines = [ASS_HEADER]

    for seg in segments:
        start = format_ass_time(seg["start"])
        end = format_ass_time(seg["end"])
        original_jp = sanitize_for_ass(seg["text"])

        if not original_jp:
            continue

        romaji_text = sanitize_for_ass(
            romanize_with_tokens(original_jp, sudachi_tokenizer, split_mode, kks)
        )

        # Comentario com o japones original (kanji/kana), so para QC humano.
        lines.append(f"Comment: 0,{start},{end},Default,,0,0,0,,[JP] {original_jp}")
        # Linha real exibida, com o romaji, na mesma janela de tempo do segmento.
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{romaji_text}")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"[+] Arquivo .ass gerado em: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Transcreve audio JP com Whisper, tokeniza com SudachiPy, romaniza e gera .ass para QC humano."
    )
    parser.add_argument("audio", help="Caminho do arquivo de audio/video de entrada.")
    parser.add_argument("-o", "--output", default=None, help="Caminho do .ass de saida (padrao: mesmo nome do audio).")
    parser.add_argument(
        "--model",
        default="large",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Tamanho do modelo Whisper (padrao: medium, recomendado para RTX 3050 8GB).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Dispositivo para rodar o Whisper (padrao: cuda).",
    )
    parser.add_argument(
        "--sudachi-mode",
        default="C",
        choices=["A", "B", "C"],
        help="Granularidade da tokenizacao do Sudachi. A=morfemas miudos, C=palavras compostas (padrao: C).",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[!] Arquivo nao encontrado: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(audio_path.with_suffix(".ass"))

    segments = transcribe(str(audio_path), args.model, args.device)
    build_ass(segments, output_path, args.sudachi_mode)


if __name__ == "__main__":
    main()
