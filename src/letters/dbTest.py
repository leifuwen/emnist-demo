from tinydb import TinyDB, Query

db = TinyDB('database/letters.json')

tb = db.table('letters')
# count = 1
# for i in range(1, 21):
#     if i <= 10:
#         if i % 2 == 1:
#             index = table.insert({"answer": "A"})
#         else:
#             index = table.insert({"answer": "C"})
#     else:
#         if i % 2 == 1:
#             index = table.insert({"answer": "B"})
#         else:
#             index = table.insert({"answer": "D"})
Q = Query()
print(tb.get(doc_id=1)['answer'] == 'A')

db.close()
