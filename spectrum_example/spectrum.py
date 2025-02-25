from bs4 import BeautifulSoup

file_name = 'MTBLS36_20100920_04_NOR_25.mzML'

# 读取XML文件
with open(file_name, 'r') as f:
    xml_data = f.read()

# print(xml_data)

# 创建BeautifulSoup对象
soup = BeautifulSoup(xml_data, 'xml')

spectrums = soup.find_all('spectrum')
for spectrum in spectrums:
    print('Index: ', spectrum.get('index'))
    print('Scan: ', spectrum.get('id').split('=')[-1])

    cvparams = spectrum.find_all('cvParam')

    for cvparam in cvparams:
        name = cvparam.get('name')
        accession = cvparam.get('accession')

        if name == 'MS1 spectrum':
            print('MS1 spectrum value: ', cvparam.get('value'))

        elif name == 'ms level':
            print('ms level value: ', cvparam.get('value'))

        elif name == 'base peak m/z':
            print('base peak m/z value: ', cvparam.get('value'))

        elif name == 'base peak intensity':
            print('base peak intensity value: ', cvparam.get('value'))

        elif name == 'scan start time':
            print('scan start time value: ', cvparam.get('value'))

        if accession == 'MS:1000130' or accession == 'MS:1000129':
            print('scan name: ', cvparam.get('name'))
    print('------')  # separator for different spectrums

print(len(spectrums))