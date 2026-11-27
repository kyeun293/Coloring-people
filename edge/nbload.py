"""
nbload.py: 노트북 소스(edge_coloring.py)의 0~3장(설정·데이터·검출기·후처리·색칠판 함수)만 불러오기

스크립트(run_photo.py, make_dev_set.py, dev_eval.py)가 노트북과 똑같은 코드를 쓰도록 하는 용도.
예시 그림 셀(RUN_DEMO)과 4-2 이후(평가 실행/실험)는 실행하지 않음. 평가 지표 함수(4장 앞부분)는 포함.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_SRC = os.path.join(HERE, 'edge_coloring.py')


def load_nb():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    src = open(NB_SRC, encoding='utf-8').read()
    marker = '# ── 4-2. 평가 실행'
    if marker not in src:
        raise RuntimeError(f'{NB_SRC}에서 "{marker}" 셀을 찾을 수 없음')
    src = src.split(marker)[0]
    ns = {'__name__': 'edge_nb', '__file__': NB_SRC, 'RUN_DEMO': False}
    exec(compile(src, NB_SRC, 'exec'), ns)
    return ns
