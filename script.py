data = []
with open("C:/Users/guilh/Desktop/facial/clustering-de-images/teste.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        line.strip()
        line = " ".join(line.split("-")[3:])
        data.append(line)
indentificacao = sum([int(d.strip()[0]) for d in data if "iden" in d])
# acertos = sum([int(d.strip().split(" ")[2]) for d in data if " incorreta" in d])
s = 0
for d in data :
    d = " ".join(d.split(" ")[5:8])
    if " correta" in d:
        s += int(d[0])
print(indentificacao)
print(s)
print(s/indentificacao)