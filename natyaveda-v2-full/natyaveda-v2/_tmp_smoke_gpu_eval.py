import json
import re
from pathlib import Path
from src.inference.predictor import Predictor

root = Path('.')
smoke_dir = root / 'data' / 'smoke test'
out_dir = root / 'outputs' / 'smoke_test_gpu'
out_dir.mkdir(parents=True, exist_ok=True)

predictor = Predictor(
    checkpoint_path=str(root / 'weights' / 'danceformer_best.pt'),
    pose_model='rtmw-x',
    device='cuda',
    clip_length=64,
)

rows = []
for video in sorted(smoke_dir.glob('*.mp4')):
    m = re.match(r'auto_([^_]+)_', video.name)
    expected = m.group(1).lower() if m else 'unknown'
    result = predictor.predict_video(
        video_path=video,
        output_video_path=None,
        show_skeleton=False,
        show_mudras=False,
    )
    pred = result.get('dance_form', 'error')
    conf = float(result.get('confidence', 0.0))
    ok = (pred == expected)
    row = {
        'video': video.name,
        'expected': expected,
        'predicted': pred,
        'confidence': conf,
        'ok': ok,
    }
    rows.append(row)
    with open(out_dir / f"{video.stem}.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

correct = sum(1 for r in rows if r['ok'])
total = len(rows)
print('\nSMOKE_GPU_CLASSIFICATION_REPORT')
print(f'ACCURACY {correct}/{total} = {correct/total if total else 0:.3f}')
for r in rows:
    status = 'OK' if r['ok'] else 'MISS'
    print(f"{status} | expected={r['expected']:<12} predicted={r['predicted']:<12} conf={r['confidence']*100:5.1f}% | {r['video']}")
