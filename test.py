from openmodeldb import OpenModelDB

db = OpenModelDB()

# compacts = db.find(scale=2, architecture="compact")

# for compact in compacts:
#     print(compact)

# model = db.search("GT-v2-evA")

# print(model)

# print("AnimeJaNai_HD_V3Sharp1_Compact" in db)

# db.download("GT-v2-evA", format="onnx")
db.test_integrity("downloads/2x-GT-v2-evA.onnx")