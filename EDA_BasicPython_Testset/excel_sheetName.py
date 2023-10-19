#
# from openpyxl import load_workbook
# wb = load_workbook("20221213_dataprocessing3.xlsx") #打开工作簿
# sheet_names=wb.get_sheet_names() #获得工作簿的所有工作表名
# for sheet_name in sheet_names: #遍历每个工作表，并将每个工作表名称改成新的
#     ws=wb[sheet_name]
#     # ws.title=sheet_name.replace("-","")
#     ws.title = sheet_name.replace("+", "")
# wb.save('20221213_test.xlsx')