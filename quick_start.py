"""Windows-friendly single entry point for local Qwen + .env chat/vision model.

Nothing is sent to the remote API until the user explicitly starts training
or submits text/image to analyse. Secrets stay in .env, not in the ZIP.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
TRAINED = ROOT / 'model_hybrid_qwen' / 'thresholds.json'


def ask(prompt, default=''):
    try:
        answer = input(prompt).strip()
    except EOFError:
        return default
    return answer or default


def load_config():
    values = {}
    env = ROOT / '.env'
    if not env.is_file():
        return values
    for raw in env.read_text(encoding='utf-8-sig').splitlines():
        raw = raw.strip()
        if raw and not raw.startswith('#') and '=' in raw:
            key, value = raw.split('=', 1)
            values[key.strip()] = value.strip().strip('"\'')
    return values


def configured(values):
    key = values.get('API_KEY', '').strip()
    return (bool(key) and key.lower() not in
            ('sk-...', 'sk-xxx', 'your_api_key_here', 'changeme', 'paste_your_api_key_here')
            and bool(values.get('MODEL', '').strip())
            and bool(values.get('API_URL', '').strip()))


def configure_env():
    env = ROOT / '.env'
    if not env.exists():
        shutil.copyfile(ROOT / '.env.example', env)
        print('Создан .env из шаблона. Вписать нужно только API_KEY, если API_URL/MODEL подходят.')
    values = load_config()
    if configured(values):
        print('OK: .env найден, API_KEY, API_URL и MODEL указаны (сам ключ не выводится).')
        return True
    print('Открой .env в Блокноте, впиши API_KEY, сохрани Ctrl+S и вернись сюда.')
    if sys.platform == 'win32':
        try:
            subprocess.Popen(['notepad.exe', str(env)], cwd=ROOT)
        except OSError:
            pass
    ask('После сохранения .env нажми Enter... ')
    values = load_config()
    if not configured(values):
        print('Не вижу заполненных API_KEY / API_URL / MODEL в .env. Проверь файл и запусти снова.')
        return False
    print('OK: настройки LLM прочитаны.')
    return True


def run_local(*args):
    cmd = [PYTHON, str(ROOT / 'local_qwen.py'), *args]
    print('\n>>>', ' '.join(x for x in cmd[2:] if x != '--llm-from-env'), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def is_new_model_ready():
    if not TRAINED.is_file():
        return False
    try:
        settings = json.loads(TRAINED.read_text(encoding='utf-8'))
        return settings.get('with_llm_trained') is True and settings.get('llm_segmentation') is True
    except (OSError, ValueError):
        return False


def main():
    print('\n=== Kirusha AI Detector: запуск ===')
    if not configure_env():
        return 2
    print('\nПроверяю локальный Qwen (LM Studio, порт 1234)...')
    if run_local('check'):
        print('\nНе удалось подключиться к Qwen. Запусти LM Studio, включи Local Server')
        print('на http://127.0.0.1:1234 и загрузи embedding-модель под именем qwen3-embedding-local.')
        return 3
    if not is_new_model_ready():
        print('\nНовая гибридная модель ещё не обучена. Потребуются запросы к внешней')
        print('LLM из .env для сегментации и обучения. Возможно расходование квоты API.')
        if ask('Начать первичное обучение сейчас? [Д/н]: ', 'д').lower() in ('н', 'n', 'no', 'нет'):
            print('Чтобы обучить позже, запусти 09_QWEN_ENV_LLM_TRAIN.bat.')
            return 0
        if run_local('train', '--llm-from-env', '--with-llm', '--llm-segmentation'):
            print('\nОбучение завершилось с ошибкой — смотри вывод выше. Модель не готова.')
            return 4
        if not is_new_model_ready():
            print('Обучение не сохранило полноценную with_llm модель.')
            return 5
    print('\nМодель готова. Qwen работает локально; текст/фото в гибридном режиме')
    print('передаются внешнему провайдеру из .env. Без фото vision не вызывается.')
    while True:
        print('\n1 — проверить текст/SMS')
        print('2 — проверить текст + фотографию')
        print('3 — проверить EML-письмо (с автоматическим поиском фото)')
        print('0 — выход')
        choice = ask('Выбери: ', '0')
        if choice == '0':
            break
        if choice == '1':
            run_local('text', '--llm-from-env')
        elif choice == '2':
            photo = ask('Путь к фото JPG/PNG/WEBP (можно перетащить файл в окно): ').strip('"')
            if not photo or not Path(photo).is_file():
                print('Файл не найден — выбери 1 для проверки только текста.')
            else:
                run_local('text', '--llm-from-env', '--image', photo)
        elif choice == '3':
            eml = ask('Путь к .eml: ').strip('"')
            if not Path(eml).is_file():
                print('Файл не найден.')
            else:
                run_local('analyze', '--llm-from-env', eml)
        else:
            print('Нет такого пункта.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nОстановлено пользователем.')
        raise SystemExit(130)
