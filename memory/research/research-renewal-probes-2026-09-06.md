# Алгебраические пробы текущей сонификации
Дата: 2026-09-06. Статус: исследовательские наблюдения.
Связанный [отчёт](research-renewal-2026-09-06.md).

## Что проверено

Взяты неизменённые тела функций _select_modes и map_laplacian из casynth_core.py. Построение разреженной матрицы и операция графового Лапласиана заменены эквивалентными плотными операциями NumPy для небольших симметричных графов.

Причина: в доступном bundled Python есть NumPy, но нет SciPy; обычный импорт ядра завершился ModuleNotFoundError. Это **проверка алгебры отображения**, не запуск приложения, не полный тест SciPy-пути и не прослушивание.

SHA256 casynth_core.py:
`365cc598ea6eeff596104c162afba2ad73e2f93877ebe446789b3951fadcf021`.

Связанные проверенные файлы:
- casynth_engine.py: `3fc9588732af88b91466dde7efba9bb614840f69bafabe3ae46ae8f0a030e8f1`;
- casynth_config.py: `f4ac882cabe65f347e5d4e31f819b82a887d7d5afe5746ceef8ae8df251f6da7`.

Эти отпечатки обозначают прочитанную рабочую копию. В репозитории до исследования уже были пользовательские изменения в других файлах.

После создания отчётов код воспроизведения из этого приложения повторно исполнен: все численные поля совпали с записанными результатами. Локальные ссылки трёх документов проверены. Обязательная команда check.py также была вызвана; проверки приложения не смогли начаться из-за отсутствия SciPy и pygame в доступном runtime. Исходники синтезатора не изменены, прохождение регрессий не заявляется.

## Результаты

### P1. Разная геометрия, одинаковый граф

Прямая:
```text
1111
```

Ломаная:
```text
1100
0010
0001
```

Для 8-соседства обе формы дают одну и ту же матрицу пути из четырёх вершин при используемом порядке обхода. Полные частоты и амплитуды map_laplacian совпадают при shape=0 и shape=1. Таким образом, граница класса неразличимых форм здесь задана уже представлением графа.

Это не означает, что любой изгиб должен обязательно менять звук. Но если исследователь собирается озвучивать геометрические различия, ему нужно знать, какие из них его представление вообще сохраняет.

### P2. Число мод не равно числу разных частот

Пара клеток, треугольник K3 и блок 2×2 дают 1, 2 и 3 ненулевые моды соответственно, но после привязки к f0 все частоты равны f0.

Для постоянных фаз и амплитуд:
`sum_k a_k*sin(omega*t+phi_k) = A*sin(omega*t+Phi)`.
Поэтому установившаяся сумма остаётся одночастотной. Значения A могут отличаться; атаки, огибающие, хвосты и история приложения здесь не проверялись.

### P3. Порог децимации и локальность

Блок 16×16 содержит 256 клеток. После добавления одной клетки (16,0) порог выбирает шаг 2, и остаётся 65 узлов.

Вторая ненулевая частота меняет отношение к f0 с 1 до 1.041706; восьмая — с 2.722100 до 2.480791. Это численное свидетельство дискретной смены представления. Слышимость и музыкальная значимость должны проверяться отдельно.

### P4. Потолок числа узлов

Маска 33×33: все чётные строки и все чётные столбцы живы. Это связный объект из 833 клеток. Формула выбирает шаг 2, после чего остаётся полный блок 17×17, то есть 289 узлов. Заявленный максимум 256 не соблюдается.

Это отдельный технический дефект ограничения стоимости. Он не объясняет автоматически бедность палитры.

### P5. Буквальное чтение живых клеток бинарной формы

Если брать значения бинарной маски только в живых клетках, последовательность состоит из единиц. После удаления DC все Fourier-компоненты равны нулю.

Это проверка буквального возможного прочтения fallback в предложении scan, а не баг реализованного scan: такого движка в текущем реестре нет. Спецификацию нужно уточнить до реализации.

## Машинный вывод пробы

```json
{
  "method": "Unmodified mapping functions extracted via AST; sparse construction and graph Laplacian replaced with equivalent dense numpy operations. This is an algebra probe, not an application regression test.",
  "path4_shape_0.0": {
    "max_freq_difference": 0,
    "max_amp_difference": 0,
    "ratios": [
      1,
      1.84775907,
      2.41421356
    ]
  },
  "path4_shape_1.0": {
    "max_freq_difference": 0,
    "max_amp_difference": 0,
    "ratios": [
      1,
      1.84775907,
      2.41421356
    ]
  },
  "pair": {
    "frequencies": [
      220
    ],
    "amplitudes": [
      1
    ]
  },
  "triangle": {
    "frequencies": [
      220,
      220
    ],
    "amplitudes": [
      1,
      0.5
    ]
  },
  "block": {
    "frequencies": [
      220,
      220,
      220
    ],
    "amplitudes": [
      1,
      0.5,
      0.33333333
    ]
  },
  "before_one_bit": {
    "live": 256,
    "stride": 1,
    "kept": 256,
    "ratios_first_8": [
      1,
      1,
      1.378787,
      1.981126,
      1.990168,
      2.163697,
      2.163697,
      2.7221
    ]
  },
  "after_one_bit": {
    "live": 257,
    "stride": 2,
    "kept": 65,
    "ratios_first_8": [
      1,
      1.041706,
      1.339529,
      1.831056,
      2.043956,
      2.064505,
      2.133219,
      2.480791
    ]
  },
  "node_cap_counterexample": {
    "live": 833,
    "stride": 2,
    "kept": 289,
    "cap": 256
  },
  "live_only_binary_scan": {
    "ac_fft_max": 0
  }
}
```

## Воспроизведение

Запускать из корня CASynth-2 с Python и NumPy. Код только читает исходник и печатает результат. Он не изменяет ядро. После изменения исходника результаты относятся уже к новой версии.

```python
import ast, json, pathlib
import numpy as np

source = pathlib.Path("casynth_core.py").read_text(encoding="utf-8")

class Dense:
    def __init__(self, a):
        self.a = a
    def toarray(self):
        return self.a

def csr_matrix(arg, shape):
    data, (rows, cols) = arg
    a = np.zeros(shape)
    np.add.at(a, (rows, cols), data)
    return a

def sparse_laplacian(a):
    return Dense(np.diag(a.sum(axis=0)) - a)

ns = dict(np=np, csr_matrix=csr_matrix,
          sparse_laplacian=sparse_laplacian,
          N_PARTIALS_DEFAULT=20, MAX_LAPLACIAN_NODES=256,
          _GUARD=0.45 * 44100)

for item in ast.parse(source).body:
    if isinstance(item, ast.FunctionDef) and item.name in {
        "_select_modes", "map_laplacian"
    }:
        exec(compile(ast.Module(body=[item], type_ignores=[]),
                     "casynth_core.py", "exec"), ns)

mapping = ns["map_laplacian"]

def mask(coords):
    a = np.zeros((max(r for r, c in coords) + 1,
                  max(c for r, c in coords) + 1), np.uint8)
    for r, c in coords:
        a[r, c] = 1
    return a

out = {}
p = mask([(0, 0), (0, 1), (0, 2), (0, 3)])
q = mask([(0, 0), (0, 1), (1, 2), (2, 3)])
for shape in [0.0, 1.0]:
    f, a = mapping(p, 220, n=20, fullshape=True, shape=shape)
    g, b = mapping(q, 220, n=20, fullshape=True, shape=shape)
    out["path4_shape_" + str(shape)] = dict(
        max_freq_difference=float(np.max(abs(f-g))),
        max_amp_difference=float(np.max(abs(a-b))),
        ratios=(f[f > 0]/220).round(8).tolist())

for name, coords in [
    ("pair", [(0, 0), (0, 1)]),
    ("triangle", [(0, 0), (0, 1), (1, 1)]),
    ("block", [(0, 0), (0, 1), (1, 0), (1, 1)])
]:
    f, a = mapping(mask(coords), 220, n=20,
                   fullshape=True, shape=1.0)
    out[name] = dict(
        frequencies=f[f > 0].round(8).tolist(),
        amplitudes=a[f > 0].round(8).tolist())

p = np.zeros((17, 16), np.uint8)
p[:16, :] = 1
q = p.copy()
q[16, 0] = 1
for name, a in [("before_one_bit", p), ("after_one_bit", q)]:
    count = int(a.sum())
    stride = int(np.ceil(np.sqrt(count/256))) if count > 256 else 1
    f, _ = mapping(a, 220, n=20, fullshape=True)
    out[name] = dict(live=count, stride=stride,
                     kept=int(a[::stride, ::stride].sum()),
                     ratios_first_8=(f[:8]/220).round(6).tolist())

y, x = np.indices((33, 33))
p = ((y % 2 == 0) | (x % 2 == 0)).astype(np.uint8)
stride = int(np.ceil(np.sqrt(int(p.sum())/256)))
out["node_cap_counterexample"] = dict(
    live=int(p.sum()), stride=stride,
    kept=int(p[::stride, ::stride].sum()), cap=256)
out["live_only_binary_scan"] = dict(
    ac_fft_max=float(np.max(abs(np.fft.rfft(np.ones(16)))[1:])))
print(json.dumps(out, indent=2))
```

## Граница интерпретации

Проверено наличие точных совпадений и механизмов скачка. Не измерялись их распространённость в реальных сессиях, психоакустическая дистанция, слуховое предпочтение или вклад относительно аудиодвижка. Для этого нужен R0 из проекта экспериментов.
