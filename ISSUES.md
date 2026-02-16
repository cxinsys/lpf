# LPF Known Issues

> 작성일: 2026-02-17
> 대상: 무당벌레 날개 색상 패턴 형성 시뮬레이션 (반응-확산 PDE 프레임워크)

전반적인 아키텍처 설계(Factory 패턴, Strategy 패턴, 다중 백엔드 지원)는 잘 되어 있으나,
코드 품질 면에서 상당수의 버그와 개선점이 발견되었습니다.

---

## 1. CRITICAL 버그 (런타임 크래시)

- [x] **1-1. `MaxHistogramRootMeanSquareError` 잘못된 상속**
  - **위치:** `lpf/objectives/histrmse.py:84`
  - **내용:** `EachHistogramRootMeanSquareError` 대신 `Objective`를 상속하여, `super().compute()`가 `NotImplementedError`를 발생시킵니다.

- [x] **1-2. DOPRI5 solver가 NumPy/CuPy/JAX에서 작동 불가**
  - **위치:** `lpf/solvers/dopri5solver.py:117`
  - **내용:** `model.am.sqrt()`, `model.am.mean()`은 `TorchModule`에만 정의되어 있습니다. 다른 백엔드에서는 `AttributeError`가 발생합니다.

- [x] **1-3. AdaptiveRK45 solver가 NumPy 전용**
  - **위치:** `lpf/solvers/adaptiverk45solver.py:63-97`
  - **내용:** `np.zeros`, `np.sum`, `np.abs`, `np.max` 등을 하드코딩하여 GPU 백엔드(CuPy, PyTorch, JAX)에서 사용 불가합니다.

- [x] **1-4. Adaptive solver의 시간 적분 오류**
  - **위치:** `lpf/solvers/dopri5solver.py`, `lpf/solvers/adaptiverk45solver.py`
  - **내용:** adaptive sub-step이 `dt`보다 작을 때 한 번만 sub-step을 수행하지만, 외부 루프는 full `dt`만큼 시간을 전진시킵니다. 시간 불일치로 인해 적분 결과가 부정확합니다.

- [x] **1-5. `SolverFactory`에서 `adaptive_rk45`가 잘못된 solver 생성**
  - **위치:** `lpf/solvers/solverfactory.py:42`
  - **내용:** `"rk45" in "adaptiverk45"`가 `True`이므로, `AdaptiveRKF45Solver` 대신 `RungeKuttaSolver`(고정 스텝 RK4)가 반환됩니다.

- [x] **1-6. `colorize()`에서 numpy 배열 truthiness 오류**
  - **위치:** `lpf/models/twocomponentmodel.py:240`
  - **내용:** `if not thr_color:`에서 `thr_color`가 다중 원소 numpy 배열이면 `ValueError` 발생. `if thr_color is None:`이어야 합니다.

- [x] **1-7. `is_early_stopping()`에서 미정의 속성 참조**
  - **위치:** `lpf/models/twocomponentmodel.py:231-232`
  - **내용:** `self._f`, `self._g`는 어디에서도 할당되지 않습니다. `reactions()`가 로컬 변수로만 `(f, g)`를 반환하고 인스턴스에 저장하지 않아 `AttributeError`가 발생합니다.

- [x] **1-8. JaxModule의 여러 메서드가 존재하지 않는 API 사용**
  - **위치:** `lpf/array/module.py:639`, `lpf/array/module.py:659-663`
  - **내용:** `jnp.generic` 미존재, `clear_memory()`에서 `jax.numpy.clear_backends()` 등 존재하지 않는 메서드 호출

- [x] **1-9. CupyModule `clear_memory()`에서 `self._module.dev` 미존재**
  - **위치:** `lpf/array/module.py:244`
  - **내용:** `cupy`에 `dev` 속성이 없어 `AttributeError` 발생.

- [x] **1-10. Converter에서 누락된 쉼표로 문자열 합쳐짐**
  - **위치:** `lpf/converters/gierermeinhardtconverter.py:16-17`
  - **내용:** `"nu"` `"u0"` 사이에 쉼표가 없어 Python이 `"nuu0"`으로 연결합니다. 파라미터 이름 리스트가 8개가 아닌 7개가 되어 인덱스 불일치가 발생합니다.

---

## 2. 의미적 버그 (잘못된 결과를 조용히 생산)

- [x] **2-1. `not x` 패턴의 광범위한 오용 (40+ 곳)**
  - 코드 전반에서 `if not x:`를 `if x is None:` 대신 사용합니다. `x=0`, `x=0.0`, `x=""`, `x=[]` 등 falsy 값이 의도치 않게 기본값으로 대체됩니다.
  - **영향 받는 주요 위치:**
    - `twocomponentmodel.py` — `width=0`, `height=0`, `dx=0`, `index=0`, `generation=0`, `fitness=0.0` 등
    - `solver.py` — `dt=0.0`, `rtol=0.0`, `n_iters=0`
    - 모든 objective 파일 — `coeff=0`
    - `evosearch.py:121` — `generation=0`이 빈 문자열로 처리됨

- [x] **2-2. `to_dict()`에서 유효한 0값 누락**
  - **위치:** `lpf/models/twocomponentmodel.py:352-358`
  - **내용:** `if index:`, `if generation:`, `if fitness:`가 0값을 건너뜁니다. `index=0`(첫 번째 배치)은 직렬화되지 않습니다.

- [x] **2-3. `float16` 캐스팅으로 인한 잘못된 유효성 검사**
  - **위치:** `lpf/models/twocomponentmodel.py:219`, `lpf/utils/validation.py:12`
  - **내용:** float16 최대값은 65504이므로, 65504~inf 범위의 유효한 float32 값이 `inf`로 변환되어 잘못된 상태로 판정됩니다.

- [x] **2-4. `TorchModule.repeat()`의 시맨틱 불일치**
  - **위치:** `lpf/array/module.py:577-582`
  - **내용:** `axis=None`일 때 PyTorch의 `repeat`은 NumPy와 다르게 동작합니다. NumPy는 원소 반복 `[a,a,b,b]`, PyTorch는 타일링 `[a,b,a,b]`.

- [x] **2-5. `solver.trj_y` 속성이 잘못된 위치에서 읽음**
  - **위치:** `lpf/solvers/solver.py:40-41`
  - **내용:** trajectory는 `self._trj_y`에 저장되지만, property는 `self._model.trj_y`를 반환합니다.

- [x] **2-6. Pattern 저장이 morph 저장에 종속**
  - **위치:** `lpf/solvers/solver.py:190-202`
  - **내용:** `dpath_pattern` 체크가 `if dpath_morph:` 블록 안에 중첩되어, morph 없이 pattern만 저장하는 것이 불가능합니다.

---

## 3. YAML 설정 버그

- [ ] `config/evosearch/config_search_ahexaspilota_gpu.yaml:16` — `COLOV_V` 오타 → `COLOR_V`여야 함
- [ ] `config/evosearch/config_search_all_cpu.yaml:11` — `NUM_INIT_PTS` → `N_INIT_PTS`여야 함 (KeyError 발생)
- [ ] `config/evosearch/config_search_spectabilis_gpu.yaml:17` — `INIT_POP`이 spectabilis가 아닌 `init_pop_axyridis`를 참조
- [ ] CPU config 파일들 — `MODEL`, `SOLVER` 키 누락 → KeyError 발생

---

## 4. 이름 짓기 / 복사-붙여넣기 오류

- [ ] `lpf/models/schnakenbergmodel.py:7-11` — docstring이 "Gierer-Meinhardt model"로 잘못 기술
- [ ] `lpf/models/schnakenbergmodel.py:143` — `# end of class GrayScottModel` (잘못된 클래스명)
- [ ] `lpf/models/gierermeinhardtmodel.py:143` — `# end of class GrayScottModel` (동일)
- [ ] `lpf/converters/gierermeinhardtconverter.py:4` — 클래스명이 `GiererMeinhardtModel` (→ `GiererMeinhardtConverter`여야 함)
- [ ] `lpf/converters/schnakenbergconverter.py:4` — 클래스명이 `SchnakenbergModel` (→ `SchnakenbergConverter`여야 함)
- [ ] `lpf/objectives/perceptualsimilarity.py:10` — "FrechetInceptionDistance" 오류 메시지 (LPIPS 파일인데)
- [ ] `lpf/models/diploidy.py:51` — `"maternal_model.parms"` 오타 → `"params"`
- [ ] 모든 `parse_params` — `@classmethod`인데 첫 파라미터가 `self` (→ `cls`여야 함)

---

## 5. 프로젝트 인프라 문제

- [x] **5-1. 테스트 완전 부재** — 회귀 테스트 추가 완료
- [ ] **5-2. CI/CD 미설정** — `.github/workflows/`, `.travis.yml`, `tox.ini` 등 자동화된 품질 보증 인프라가 전혀 없습니다.
- [ ] **5-3. 패키징 문제**
  - `setup.py`만 사용 (`pyproject.toml` 없음, PEP 517/518 미준수)
  - `install_requires` 미선언 — `pip install .`로 설치 시 의존성 0개 설치
  - `package_data` 미설정 — `lpf/data/`의 이미지 파일들이 배포에 포함되지 않음
  - `__version__` 미정의
- [ ] **5-4. 의존성 관리**
  - `requirements.txt`에 버전 핀 없음
  - PyTorch가 어디에도 의존성으로 선언되지 않음
  - `moviepy`가 `video.py`에서 import되지만 requirements에 없음
  - `imghdr` 모듈이 Python 3.13에서 제거됨
- [ ] **5-5. 쉘 스크립트 다수가 깨져 있음** — `run_evosearch_all.sh` 등이 존재하지 않는 Python 파일을 참조합니다.

---

## 6. 설계/아키텍처 개선점

- [ ] **6-1. `not x` → `x is None`으로 전면 교체 필요** — 가장 광범위한 버그 패턴 (40곳 이상)
- [ ] **6-2. 모델 간 대량 코드 중복** — `to_dict()`, `parse_params()`, `get_param_bounds()`가 거의 동일. 파라미터 이름/범위를 클래스 속성으로 선언하면 수백 줄 절약 가능.
- [ ] **6-3. Factory에서 substring 매칭의 취약성** — `if "x" in name:` 패턴을 정확 매칭 딕셔너리로 교체 필요
- [ ] **6-4. `ConverterFactory`가 Liaw만 지원** — 다른 Converter와 `TwoComponentCrosstalkDiploidModel`이 factory에 미등록
- [ ] **6-5. VGG16 모델을 4회 중복 로드** (`lpf/objectives/vggperceptualloss.py:29-32`)
- [ ] **6-6. 진화 탐색의 무제한 캐시 증가** (`lpf/search/evosearch.py:91`) — LRU 캐시 등으로 교체 필요
- [ ] **6-7. `image.py`에서 하드코딩된 이미지 크기** — resize/crop 좌표가 매직 넘버
- [ ] **6-8. Optional dependency import 처리 불일치** — `vggperceptualloss.py`, `perceptualsimilarity.py`
- [ ] **6-9. `rdmodel.py` `y_mesh` setter에 dead code** — `if self._y_mesh is None:` 분기 도달 불가능
- [ ] **6-10. `AdamsBashforth2Solver`의 stale state** — `_prev_dydt`가 solve() 간에 리셋 안 됨
- [ ] **6-11. `get_param_bounds()` range 계산 오류** — `range(N, 2 * n_init_pts, 2)` → `range(N, N + 2 * n_init_pts, 2)`

---

## 7. 기타 사소한 이슈

- [ ] `module.py:46,68` — Exception 생성자에 tuple 전달 (f-string 사용해야 함)
- [ ] `module.py:74,76` — `== None` 대신 `is None` 사용해야 함
- [ ] `solverfactory.py:1-19` — 주석 처리된 dead code
- [ ] `solverfactory.py:72` — stiff 문제에 explicit method 추천 (implicit method 필요)
- [ ] `rungekuttasolver.py:5` — docstring이 "RK45"라고 기술하지만 실제로는 RK4
- [ ] `adamsbashforth2solver.py:25` — `_prev_t` 저장하지만 어디에서도 읽지 않음
- [ ] `image.py:94,244,379`, `video.py:13` — 오타: "doest not exists" → "does not exist"
- [ ] `video.py:3` — deprecated `imghdr` 모듈 사용 (Python 3.13에서 제거됨)
- [ ] `video.py:17-27` — 비디오 프레임이 정렬되지 않음 (`os.listdir` 결과 미정렬)
- [ ] `diploidy.py:42` — `id(x) == id(y)` 대신 `x is y` 사용해야 함
- [ ] `.gitignore` — Django, Flask 등 관련 없는 항목 포함
- [ ] `search/run_evosearch_succinea.psh` — 잘못된 파일 확장자 `.psh`
- [ ] 쉘 스크립트에 shebang (`#!/bin/bash`) 누락

---

## 8. 우선순위 요약

| 우선순위 | 카테고리 | 항목 수 | 상태 |
|---|---|---|---|
| **P0 - 즉시 수정** | 런타임 크래시 버그 | 10개 | **완료** |
| **P1 - 높음** | 잘못된 결과 생산 버그 | 6개 | **완료** |
| **P2 - 중간** | 설정/이름짓기 오류, 인프라 | ~15개 | 미해결 |
| **P3 - 낮음** | 코드 품질/설계 개선 | ~15개 | 미해결 |
