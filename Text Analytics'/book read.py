


def read_book(title_path):
    with open(title_path,'r',encoding = 'utf8') as current_file:
        text = current_file.read()
        text = text.replace('\n','').replace('\r','')
    return text