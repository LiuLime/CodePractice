import zlib
import base64
import numpy as np

from bs4 import BeautifulSoup


def convert_to_float_array(uncompressed_data):
    float_array = np.frombuffer(uncompressed_data, dtype=np.float64)
    return float_array


# file_name = 'MTBLS36_20100920_04_NOR_25.mzML'
# file_name = './test_dataset/MTBLS5568_20180205_TOFI2_B2_HILIC_POS_097.mzML'
file_name = './test_dataset/MTBLS5568_20180205_TOFI2_B2_HILIC_POS_097.mzML'
# 读取XML文件
with open(file_name, 'r') as f:
    xml_data = f.read()


def read_rt_intensity(xml_data):
    # 创建BeautifulSoup对象
    soup = BeautifulSoup(xml_data, 'lxml')

    # find single chromatogram
    chromatograms = soup.find_all('chromatogram')

    for chromatogram in chromatograms:
        # print(chromatogram)
        binaries = chromatogram.find_all('binary')
        for binary in binaries:
            # print(binary.text)
            compressed_data = binary.text
            # step 1
            decoded_data = base64.b64decode(compressed_data)
            # print(decoded_data)

            # step 2

            uncompressed_data = zlib.decompress(decoded_data)
            # print(uncompressed_data)

            # 在你的代码后添加
            float_array = convert_to_float_array(uncompressed_data)
            print(float_array[0:10])
