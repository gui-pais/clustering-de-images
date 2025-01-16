from collections import Counter

data = []
with open("resultados gerais c cluster.txt", "r",encoding="utf-8") as f:
    for line in f.readlines():
        data.append(line.strip())
        
count_detected = [int(d[0]) for d in data if "identificacao" in d]
count_detected_with_out_1 = [int(d[0]) for d in data if "identifi" in d and d[0] != "1"]
count_face = [int(d[0]) for d in data if "face" in d]

print(f"Faces identificadas: {sum(count_detected)}")
print(f"Faces corretas detectadas: {sum(count_face)}")
print(f"Faces incorretas detectadas: {sum(count_detected_with_out_1) - sum(count_face)}")
print(f"Taxa de acertos: {sum(count_face)/sum(count_detected_with_out_1):.2f}%")
print(f"Taxa de erros: {(sum(count_detected_with_out_1) - sum(count_face))/sum(count_detected):.2f}%")
print(f"Clusters com apenas uma indentificação: {sum(count_detected) - sum(count_detected_with_out_1)}" )
data = [d.split(";")[1] for d in data if ";" in d]
print(Counter(data))