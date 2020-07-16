import json

with open('./step3_data.json') as f:
    data = json.load(f)

new_saver = []
for row in data:
    if len(row[0]) != 10:
        pass
    else:
        new_saver.append(row)
with open('./step3_data.json','w') as f:
    data = json.dump(new_saver,f)