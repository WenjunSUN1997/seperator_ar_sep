import xml.dom.minidom as xmldom

class XmlProcessor():
    def __init__(self, doc_index, file_path):
        self.doc_index = doc_index
        self.file_path = file_path
        self.img_file_path = file_path.replace('xml', 'jpg')
        self.xml_obj = (xmldom.parse(file_path)).documentElement
        self.node_list = self.get_node_list()
        '''获取node列表'''

    def get_node_list(self):
        node_list = self.xml_obj.getElementsByTagName('TextRegion')
        return node_list

    def get_annotation(self):
        annotations = []
        for index, node in enumerate(self.node_list):
            textline_first = node.getElementsByTagName('TextLine')[0].getAttribute('custom')
            custom_data = textline_first.split('structure ')[-1].split(' ')[0]
            article_index = custom_data.split(':')[-1].replace(';', '')
            if article_index[0] != 'a' or article_index[-1] not in '0123456789':
                raise Exception("wrong article -d")

            reading_order = str(self.doc_index) + '_' + article_index
            bbox_str = node.getElementsByTagName('Coords')[0].getAttribute('points')
            bbox = [[int(y) for y in x.split(',')] for x in bbox_str.split(' ')]
            text = node.getElementsByTagName('TextEquiv')[-1].getElementsByTagName('Unicode')[0].childNodes[0].data
            text = text.replace('\n', '')
            text = text.replace('¬', '')

            annotations.append({'reading_order': reading_order,
                                'bbox': bbox,
                                'text': text,
                                'img_path': self.img_file_path,
                                'index': index})
        return annotations

if __name__ == "__main__":
    doc_index = 0
    file_path = '../data/AS_TrainingSet_NLF_NewsEye_v2/576454_0001_23676282.xml'
    processor = XmlProcessor(doc_index, file_path)
    print(processor.get_annotation())

