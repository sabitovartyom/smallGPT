# Первый запуск smallGPT

Команды выполняются из корня проекта. Используется существующее окружение:

```bash
source ~/torch_env/bin/activate
```

## Данные и tokenizer

Источник: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
файл `TinyStoriesV2-GPT4-train.txt`, лицензия CDLA-Sharing-1.0.
Скрипт загружает первые N полных историй, закрывая поток до загрузки всего корпуса.
В `data/raw/source.json` сохраняются revision, URL и SHA-256 исходного поднабора.
Это небольшой пилотный корпус, не случайная выборка из всего TinyStories.

```bash
python scripts/download_data.py --stories 2000
python scripts/prepare_data.py
```

Если данные уже есть, повторная загрузка не нужна. Для намеренной перезаписи
есть `--overwrite`; для отдельного эксперимента лучше другой `--output-dir`.

Подготовка удаляет точные дубликаты и делит **истории** на train/validation/test
в пропорциях 90/5/5 с seed 42, до формирования окон.
Ручной BPE учит слияния на 100 историях только из train; исходный алфавит берётся
из всего train. Размер словаря — максимум 512, включая UNK/BOS/EOS.
Эти ограничения сокращают время работы учебной Python-реализации.

```bash
# Более крупный tokenizer; 0 означает весь train. Это заметно медленнее.
python scripts/prepare_data.py --vocab-size 4096 --tokenizer-stories 0 --output-dir data/processed_large
```

Результат: `tokenizer.json`, тексты трёх split, три потока `torch.long` в `.pt`
и `metadata.json` с размерами, долями UNK и хешами. Между историями добавляются
EOS и BOS. При простом packing attention может смотреть на предыдущую историю.
Padding отсутствует. Из окна `[T+1]` получаются `x=window[:-1]` и `y=window[1:]`.

## Расширение корпуса до 20 000 историй

```bash
python scripts/download_data.py --stories 20000 --output-dir data/raw_20k --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
python scripts/prepare_data.py --input data/raw_20k/stories.jsonl --extend-from data/processed --output-dir data/processed_20k
```

При `--extend-from` сохраняются прежние tokenizer и validation/test, включая
файлы токенов. Новые уникальные истории добавляются только в train; тексты
validation/test и дубликаты исключаются. Параметры обучения tokenizer и доли
split в этом режиме не используются. Старые данные остаются в `data/processed`.
Возможные новые символы вне старого словаря становятся UNK; доля записывается
в metadata. Источник скачивается с той же revision, что и исходные 2000 историй.

```bash
python scripts/train.py --data-dir data/processed_20k --device cuda --context-length 128 --num-layers 4 --num-heads 1 --steps 50000 --warmup-steps 1000 --eval-every 1000 --output-dir artifacts/stories_20k
```

Это новый запуск с нуля. Checkpoint предыдущего запуска требует прежних данных
и размеров модели. Одновременно меняются корпус, контекст и число блоков,
поэтому этот запуск оценивает их совместное влияние.

## Tokenizer со словарём 2048

```bash
python scripts/prepare_data.py --splits-from data/processed_20k --vocab-size 2048 --tokenizer-stories 100 --output-dir data/processed_20k_vocab2048
```

`--splits-from` сохраняет исходные истории и их разбиение, но обучает новый BPE
только на train и заново кодирует все три части. Слияния учатся на 100 историях;
алфавит берётся из всего train. В отличие от `--extend-from`, прежний tokenizer
не используется. Обучение модели на новых данных запускается с
`--data-dir data/processed_20k_vocab2048` и новым `--output-dir`, без `--resume`.
Размер словаря модели определяется по подготовленному tokenizer автоматически.
Loss и perplexity разных tokenizer напрямую сравнивать нельзя: единицы
предсказания изменились. Контекст из 128 новых токенов тоже может охватывать
больше текста, чем прежний.

## Эксперимент с четырьмя головами attention

```bash
python scripts/train.py --data-dir data/processed_20k --device cuda --context-length 128 --num-layers 4 --num-heads 4 --steps 50000 --warmup-steps 1000 --eval-every 1000 --output-dir artifacts/stories_20k_heads4
```

Остальные настройки совпадают с запуском `stories_20k`. Обучение начинается
с нуля; при `--resume` число голов берётся из checkpoint, а не из аргумента CLI.
Генерация новой моделью:

```bash
python scripts/generate.py --checkpoint artifacts/stories_20k_heads4/best.pt --device cuda --prompt "Once upon a time" --max-new-tokens 200
```

Ручная реализация остаётся в `src/smallGPT/layers/attantion.py`.
В файле два самостоятельных класса: исходный `CausalSelfAttantion` с одной
головой и `MultiHeadCausalSelfAttantion` с несколькими. `TransformerBlock`
выбирает первый при `num_heads=1`, а второй при большем числе голов.
Проекции Q/K/V имеют форму `[B,T,D]`, после разделения — `[B,H,T,D/H]`.
Scores каждой головы — `[B,H,T,T]`; результат собирается обратно в `[B,T,D]`.
При D=128 и H=4 размер головы равен 32. Матрицы проекций сохраняют прежние
размеры, поэтому число обучаемых параметров не увеличивается.

## Отладка обучения на одном батче

```bash
python scripts/train.py --device cuda --overfit-batch --steps 300 --warmup-steps 10 --eval-every 50 --output-dir artifacts/overfit
```

Здесь train loss должен заметно падать; улучшения validation ждать не требуется.

## Обучение на всём train-поднаборе

```bash
python scripts/train.py --device cuda --output-dir artifacts/debug
```

Используется твой Backbone: четыре блока с четырьмя attention-головами, D=128,
контекст 128, batch size 4. `--num-layers` меняет число блоков, а не голов.
`--num-heads` задаёт число голов; embedding_dim должен делиться на него без остатка.
Точность FP32, AdamW, LR 3e-4, warmup/cosine,
clipping 1.0, 1000 optimizer updates. Матрицы имеют weight decay, bias и
LayerNorm — нет. При `--accumulation N` один update содержит N microbatches.
Объём обработанных токенов за update — `batch_size * context_length * accumulation`.

Скрипт выводит train loss только вместе с val_loss при validation и в конце.
Train loss в терминале — среднее за прошедший интервал; `--eval-every` задаёт
частоту вывода. В `metrics.jsonl` остаются метрики каждого update и дополнительный
`mean_train_loss` на шагах validation. Validation использует одинаковую отдельную выборку окон
и не изменяет веса или состояние train sampler. Test во время обучения не читается.
Текущие token embedding и LM head имеют независимые веса, как в твоём Backbone.

`last.pt` и `best.pt` содержат веса, tokenizer, конфигурацию, optimizer,
scheduler, update step, RNG и sampler state. `last.pt` сохраняется на каждом
интервале validation и в конце запуска. `best.pt` выбирается по validation.
Число блоков сохраняется в конфигурации; прежние checkpoints без этого поля
загружаются как модели с двумя блоками.
Число голов тоже сохраняется; checkpoints без `num_heads` загружаются с одной
головой, в том числе модель `stories_20k` с четырьмя блоками.

```bash
# Остановить после 50 updates, сохранив исходный бюджет и расписание на 1000.
python scripts/train.py --device cuda --stop-after 50 --output-dir artifacts/pause_demo

# Продолжить. Размеры модели и гиперпараметры берутся из checkpoint.
python scripts/train.py --device cuda --resume artifacts/pause_demo/last.pt --output-dir artifacts/pause_demo
```

Возобновление требует тех же подготовленных данных и tokenizer; хеши проверяются.
Точное совпадение траектории следует проверять на том же устройстве и окружении.

## Итоговая оценка и генерация

Test оценивай после выбора модели по validation:

```bash
python scripts/evaluate.py --checkpoint artifacts/debug/best.pt --split test --device cuda
python scripts/generate.py --checkpoint artifacts/debug/best.pt --device cuda --prompt "Once upon a time"
```

Для расширенного корпуса укажи `--data-dir data/processed_20k` при оценке и
`--checkpoint artifacts/stories_20k/best.pt` при оценке/генерации.

Оценка проходит весь test-поток и учитывает каждый следующий токен один раз,
включая последнюю короткую часть. Генерация использует temperature/top-k и EOS;
при превышении контекста берёт последние T токенов и заново нумерует позиции.
Начальное обучение на маленьком поднаборе проверяет pipeline; качество историй
нужно оценивать отдельно. Время обучения заранее не предполагается.

Для CPU замени `--device cuda` на `--device cpu`. `--device auto` выбирает
CUDA, когда она доступна этому процессу, иначе CPU.

```bash
python -B -m unittest discover -s tests -v
```
