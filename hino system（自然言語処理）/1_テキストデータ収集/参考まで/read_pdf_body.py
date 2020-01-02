import sys
from pdfminer.converter import TextConverter
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def read_body(f):
    parser = PDFParser(f)
    document = PDFDocument(parser)
    if not document.is_extractable:
        print(f'このPDF文書はテキスト抽出できません')
        return

    resource_manager = PDFResourceManager()
    device = TextConverter(resource_manager, sys.stdout, codec='utf-8')
    interpreter = PDFPageInterpreter(resource_manager, device)
    for i, page in enumerate(PDFPage.create_pages(document), 1):
        print(f"ページ: {i} {'=' * 32}")
        interpreter.process_page(page)
        print()

    device.close()

def usage():
    if len(sys.argv) < 2:
        print(f'{__file__} の後にPDFファイルを指定してください')
        sys.exit(0)

def main():
    with open(sys.argv[1], 'rb') as f:
        read_body(f)

if __name__ == '__main__':
    usage()
    main()
