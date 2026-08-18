"""
jp_transcribe_romanize.py  (v2 - com tokenizacao morfologica via SudachiPy)

Pipeline:
    Whisper (audio -> texto JP com timestamps)
        -> SudachiPy (tokenizacao + analise morfologica)
        -> leitura (reading_form) de cada token
        -> correcao manual de particulas gramaticais (ha/e/wo)
        -> romaji (via pykakasi, convertendo katakana -> hepburn)
        -> juncao dos tokens com espacos, pontuacao colada na palavra anterior

Gera um .ass com:
  - Linha "Comment" com o texto original em japones (kanji/kana), para QC humano.
    NAO aparece no player, fica so no arquivo como referencia.
  - Linha "Dialogue" logo abaixo, com o romaji, temporizada pelo mesmo segmento
    detectado pelo Whisper.

Uso:
    python jp_transcribe_romanize.py audio.mp3 --model medium --device cuda

Requisitos (ver requirements.txt):
    pip install openai-whisper pykakasi sudachipy sudachidict_core
    ffmpeg precisa estar instalado e no PATH.

Notas:
    - SplitMode do Sudachi: "C" agrupa em unidades mais proximas de "palavras"
      naturais (bom para leitura); "A" quebra em morfemas mais miudos.
      Ajustavel via --sudachi-mode.
    - Textos que o Sudachi nao reconhece (nomes estranhos, ruido, texto em
      outro idioma que o Whisper "alucinou") caem no fallback: mantem o
      surface original sem romanizar, para o QC humano identificar facil.
"""