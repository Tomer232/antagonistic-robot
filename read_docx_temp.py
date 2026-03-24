import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(path):
    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        document.close()
        tree = ET.XML(xml_content)
        # XML namespace for WordprocessingML
        word_namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        paragraphs = []
        for paragraph in tree.iter(word_namespace + 'p'):
            texts = [node.text for node in paragraph.iter(word_namespace + 't') if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        
        return '\n'.join(paragraphs)
    except Exception as e:
        return "Error reading docx: " + str(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open("docx_output.txt", "w", encoding="utf-8") as f:
            f.write(read_docx(sys.argv[1]))
        print("Success")
    else:
        print("Provide a path")
