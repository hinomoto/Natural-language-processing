import sys
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

def read_outline(f):
    parser = PDFParser(f)
    document = PDFDocument(parser, None)

    if document.catalog.get('Outlines') is not None:        #← アウトラインの有無を確認
        outlines = document.get_outlines()
        for level, title, dest, a, se in outlines:
            print(f'階層: {level}, タイトル: {title}')
    else:
        print(f'PDF文書にアウトラインはありません')

def usage():
    if len(sys.argv) < 2:
        print(f'{__file__} の後にPDFファイルを指定してください')
        sys.exit(0)

def main():
    with open(sys.argv[1], 'rb') as f:        #← コマンドライン引数で指定したファイルを開く
        read_outline(f)        #← openしたファイルオブジェクトを関数に渡す

if __name__ == '__main__':
    usage()
    main()
