"""
notebooks/odt_image_ocr_and_structured_merge.py

Скрипт/ядро для Jupyter: обработка ODT-файла в папке /Insight AI RAG-based Assistant/test_tekst
Задачи:
 1) распаковать ODT (content.xml и Pictures/)
 2) собрать DataFrame для изображений: путь_картинки, распознанный текст (OCR), метаданные
 3) собрать DataFrame для элементов документа (заголовки/параграфы) с указанием уровней
 4) визуализировать количество картинок с пустым OCR по документу
 5) показать несколько примеров для быстрой ручной оценки качества OCR
 6) "Вклеить" распознанный текст из картинок в места документа (вставлять после элемента, где картинка была)
 7) разбить документ на строгие блоки по структуре (уровень 1 -> уровень 2 -> уровень 3)

Инструкции по использованию:
 - положи этот файл в notebooks/ или запускай в ячейке Jupyter
 - проверь путь_odt = '../test_tekst/yourfile.odt' (или полный путь)
 - установи зависимости: odfpy, pytesseract, pillow, lxml, pandas, seaborn, matplotlib

Примечание: парсер content.xml построен на ElementTree с учётом пространств имён ODF.
"""

import zipfile
import os
import io
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Настройки (подправь пути) ----
PROJECT_ROOT = Path.cwd().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()
# если запускаешь прямо в корне проекта, укажи папку test_tekst:
ODT_DIR = PROJECT_ROOT / 'test_tekst'
OUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'odt_extracted'
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_OUT = OUT_DIR / 'Pictures'
IMAGES_OUT.mkdir(parents=True, exist_ok=True)

# ---- Полезные namespace ----
NS = {
    'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
    'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
    'draw': 'urn:oasis:names:tc:opendocument:xmlns:drawing:1.0',
    'xlink': 'http://www.w3.org/1999/xlink',
}

# ---- Функции ----

def extract_odt_content(odt_path: Path, out_images_dir: Path):
    """
    Распаковывает ODT и возвращает XML content (ElementTree) и список извлечённых картинок.
    Возвращаем: ElementTree root, images_list (list of (archive_name, saved_path))
    """
    z = zipfile.ZipFile(odt_path, 'r')
    content_xml = z.read('content.xml')
    root = ET.fromstring(content_xml)
    images = []
    # извлекаем картинки: все файлы в Pictures/
    for name in z.namelist():
        if name.startswith('Pictures/'):
            data = z.read(name)
            target = out_images_dir / Path(name).name
            with open(target, 'wb') as f:
                f.write(data)
            images.append((name, str(target)))
    z.close()
    return root, images


def _get_text_from_node(node):
    """Рекурсивный сбор текста из нод text:p и text:h и вложенных span/text nodes"""
    texts = []
    if node is None:
        return ''
    # node.text может быть None
    if node.text:
        texts.append(node.text)
    for child in node:
        texts.append(_get_text_from_node(child))
        if child.tail:
            texts.append(child.tail)
    return ' '.join([t for t in texts if t])


def parse_document_elements(root):
    """
    Проходит по content.xml и возвращает список элементов в порядке дельты.
    Каждый элемент: {'type': 'h'|'p'|'frame_image', 'level': int_or_None, 'text': str, 'image_ref': maybe str}
    """
    body = root.find('office:body', NS)
    if body is None:
        raise RuntimeError('content.xml does not contain office:body')
    text_elem = body.find('office:text', NS)
    elements = []
    idx = 0
    for node in text_elem:
        tag = ET.QName(node.tag).localname
        if tag == 'h':
            # heading
            level = node.attrib.get('{%s}outline-level' % NS['text']) or node.attrib.get('outline-level')
            try:
                level = int(level)
            except Exception:
                level = None
            text = _get_text_from_node(node).strip()
            elements.append({'idx': idx, 'type': 'h', 'level': level, 'text': text, 'image_refs': []})
            idx += 1
        elif tag == 'p':
            text = _get_text_from_node(node).strip()
            # check for images inside paragraph: draw:frame/draw:image
            image_refs = []
            for frame in node.findall('.//draw:frame', NS):
                img = frame.find('.//draw:image', NS)
                if img is not None:
                    href = img.attrib.get('{%s}href' % NS['xlink']) or img.attrib.get('href')
                    if href:
                        image_refs.append(href)
            elements.append({'idx': idx, 'type': 'p', 'level': None, 'text': text, 'image_refs': image_refs})
            idx += 1
        else:
            # другие теги — пропускаем или собираем как параграф
            text = _get_text_from_node(node).strip()
            if text:
                elements.append({'idx': idx, 'type': 'other', 'level': None, 'text': text, 'image_refs': []})
                idx += 1
    return elements


def ocr_image_file(image_path: str, lang='rus+eng'):
    try:
        img = Image.open(image_path)
        # можно добавить предварительную обработку: convert('L'), resize, threshold
        text = pytesseract.image_to_string(img, lang=lang)
        return text
    except Exception as e:
        print('OCR error for', image_path, e)
        return ''


# ---- Основной pipeline для одного файла ----

def process_odt_file(odt_path: Path, out_dir: Path, ocr_lang='rus+eng'):
    """Возвращает два DataFrame: df_elements и df_images"""
    root, images = extract_odt_content(odt_path, out_dir / 'Pictures')
    elements = parse_document_elements(root)

    # построим mapping href -> saved path
    href_to_path = {}
    for name, saved in images:
        # в content.xml ссылки выглядят как "Pictures/abc.png" — сохраняем как basename
        href_to_path[name] = saved
        href_to_path[Path(name).name] = saved

    # images DataFrame
    img_rows = []
    for el in elements:
        for href in el.get('image_refs', []):
            saved = href_to_path.get(href) or href_to_path.get(Path(href).name)
            if saved:
                ocr_text = ocr_image_file(saved, lang=ocr_lang)
                img_rows.append({'element_idx': el['idx'], 'href': href, 'image_path': saved, 'ocr_text': ocr_text})

    df_images = pd.DataFrame(img_rows)
    df_elements = pd.DataFrame(elements)
    df_elements['filename'] = odt_path.name

    return df_elements, df_images


# ---- Анализ качества OCR и визуализация ----

def analyze_and_visualize(df_elements, df_images, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    # сколько картинок всего и пустых по документу
    if df_images.empty:
        print('No images detected in document')
        return
    df_images['ocr_empty'] = df_images['ocr_text'].apply(lambda x: len(str(x).strip()) == 0)
    summary = df_images.groupby('ocr_empty').size().rename('count').reset_index()
    print('OCR empty summary:\n', summary)

    # визуализация: доля пустых
    counts = df_images.groupby('image_path').agg({'ocr_empty': 'sum', 'image_path': 'count'}).rename(columns={'image_path':'total'})
    counts = counts.reset_index()
    counts['empty_ratio'] = counts['ocr_empty'] / counts['total']

    plt.figure(figsize=(10,6))
    sns.histplot(df_images['ocr_text'].apply(lambda x: len(str(x).strip())==0), kde=False)
    plt.title('Count of images with empty OCR (True means empty)')
    plt.savefig(save_dir / 'ocr_empty_hist.png')
    plt.show()

    # по документу: если есть метаданные нескольких файлов, смотрим по filename
    # выводим примеры для ручной проверки: первые 5 пустых и непустых
    print('\nExamples of empty OCR images:')
    display(df_images[df_images['ocr_empty']].head()[['image_path','element_idx','ocr_text']])
    print('\nExamples of non-empty OCR images:')
    display(df_images[~df_images['ocr_empty']].head()[['image_path','element_idx','ocr_text']])

    # сохранить таблицы
    df_images.to_csv(save_dir / 'df_images.csv', index=False)
    df_elements.to_csv(save_dir / 'df_elements.csv', index=False)


# ---- Вставка OCR текста в документную структуру ----

def merge_ocr_into_elements(df_elements: pd.DataFrame, df_images: pd.DataFrame):
    """
    Для каждого image row вставляем ocr_text как дополнительный параграф в elements сразу после элемента, ссылающегося на картинку.
    Возвращает новый DataFrame df_merged, где добавлены строки типа 'ocr_insert'.
    """
    merged = []
    images_by_idx = df_images.groupby('element_idx').apply(lambda g: g.to_dict(orient='records')).to_dict() if not df_images.empty else {}

    for _, row in df_elements.sort_values('idx').iterrows():
        merged.append(dict(row))
        if row['idx'] in images_by_idx:
            for img in images_by_idx[row['idx']]:
                merged.append({'idx': f"{row['idx']}_img", 'type': 'ocr_insert', 'level': None, 'text': img['ocr_text'], 'image_refs': [img['image_path']], 'filename': row.get('filename')})
    df_merged = pd.DataFrame(merged)
    return df_merged


# ---- Построение иерархических блоков по заголовкам (levels) ----

def build_hierarchy_blocks(df_elements: pd.DataFrame):
    """
    Ожидает df_elements с колонками ['idx','type','level','text','filename'].
    Собирает блоки по уровню 1 -> уровень 2 -> уровень 3.
    Возвращает список блоков: {block_id, level1_title, level2_title, level3_title, combined_text, element_indices}
    """
    blocks = []
    current_l1 = None
    current_l2 = None
    block_id = 0

    # удостоверимся, что уровни для заголовков приведены к int или None
    df = df_elements.copy()
    # convert level column to numeric where possible
    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None
    df['level'] = df['level'].apply(to_int)

    # we'll iterate and maintain stacks
    for _, row in df.iterrows():
        if row['type'] == 'h' and row['level'] == 1:
            # start new L1 block
            current_l1 = row['text']
            current_l2 = None
            block_id += 1
            blocks.append({'block_id': block_id, 'level1_title': current_l1, 'level2_title': None, 'level3_title': None, 'combined_text': '', 'elements': [row['idx']]})
        elif row['type'] == 'h' and row['level'] == 2:
            current_l2 = row['text']
            block_id += 1
            blocks.append({'block_id': block_id, 'level1_title': current_l1, 'level2_title': current_l2, 'level3_title': None, 'combined_text': '', 'elements': [row['idx']]})
        elif row['type'] == 'h' and row['level'] == 3:
            current_l3 = row['text']
            block_id += 1
            blocks.append({'block_id': block_id, 'level1_title': current_l1, 'level2_title': current_l2, 'level3_title': current_l3, 'combined_text': '', 'elements': [row['idx']]})
        else:
            # ordinary paragraph — attach to latest block
            if blocks:
                blocks[-1]['combined_text'] += '\n' + (row['text'] if pd.notna(row['text']) else '')
                blocks[-1]['elements'].append(row['idx'])
            else:
                # no heading seen yet — create a default block
                block_id += 1
                blocks.append({'block_id': block_id, 'level1_title': None, 'level2_title': None, 'level3_title': None, 'combined_text': row['text'] or '', 'elements': [row['idx']]})
    return blocks


# ---- Утилита: показать несколько примеров для ручной оценки OCR качества ----

def show_manual_qc(df_images: pd.DataFrame, df_elements: pd.DataFrame, n=5):
    print('Manual QC examples (pairs image OCR <-> surrounding paragraph):\n')
    if df_images.empty:
        print('No images to inspect')
        return
    merged = merge_ocr_into_elements(df_elements, df_images)
    # print first n ocr_insert rows with context
    inserts = merged[merged['type'] == 'ocr_insert'].head(n)
    for _, row in inserts.iterrows():
        print('IMAGE PATH:', row['image_refs'])
        print('OCR TEXT:', row['text'][:400].replace('\n',' '))
        # find previous paragraph
        idx = row['idx']
        # idx is like '12_img' => get base
        base_idx = str(idx).split('_')[0]
        try:
            base_idx = int(base_idx)
            prev_para = df_elements[df_elements['idx'] == base_idx]['text'].values
            if len(prev_para):
                print('SURROUNDING PARA:', prev_para[0][:400])
        except Exception:
            pass
        print('-'*80)


# ---- Пример использования: обработка всех odt в папке ----

def run_for_folder(odt_folder: Path, out_dir: Path, ocr_lang='rus+eng'):
    all_elements = []
    all_images = []
    for f in odt_folder.glob('*.odt'):
        print('Processing', f.name)
        df_el, df_img = process_odt_file(f, out_dir / f.stem, ocr_lang=ocr_lang)
        df_el['source_file'] = f.name
        df_img['source_file'] = f.name
        all_elements.append(df_el)
        all_images.append(df_img)
    if all_elements:
        df_elements_all = pd.concat(all_elements, ignore_index=True)
    else:
        df_elements_all = pd.DataFrame()
    if all_images:
        df_images_all = pd.concat(all_images, ignore_index=True)
    else:
        df_images_all = pd.DataFrame()
    return df_elements_all, df_images_all


# ---- Если запускаем как скрипт, выполняем run_for_folder по умолчанию ----
if __name__ == '__main__':
    folder = Path.cwd() / 'test_tekst'
    if not folder.exists():
        # fallback если исполняем в проекте root
        folder = Path('test_tekst')
    df_elements, df_images = run_for_folder(folder, OUT_DIR, ocr_lang='rus+eng')
    print('Elements:', len(df_elements), 'Images:', len(df_images))
    analyze_and_visualize(df_elements, df_images, OUT_DIR)
    show_manual_qc(df_images, df_elements, n=8)
    df_merged = merge_ocr_into_elements(df_elements, df_images)
    blocks = build_hierarchy_blocks(df_merged)
    # сохранить
    pd.DataFrame(blocks).to_csv(OUT_DIR / 'hierarchy_blocks.csv', index=False)
    print('Saved outputs to', OUT_DIR)

    # quick print of block distribution
    from collections import Counter
    cnt = Counter([b['level1_title'] for b in blocks if b['level1_title']])
    print('Top level headings count sample:', cnt.most_common(10))

    # небольшой пример: сохранить merged text file
    with open(OUT_DIR / 'merged_document.txt', 'w', encoding='utf-8') as f:
        for row in df_merged.to_dict(orient='records'):
            f.write((str(row.get('text','')) or '') + '\n\n')

    print('Done.')
