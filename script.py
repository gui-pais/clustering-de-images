# from collections import Counter

# data = []
# with open("resultados gerais c cluster.txt", "r",encoding="utf-8") as f:
#     for line in f.readlines():
#         data.append(line.strip())
        
# count_detected = [int(d[0]) for d in data if "identificacao" in d]
# count_detected_with_out_1 = [int(d[0]) for d in data if "identifi" in d and d[0] != "1"]
# count_face = [int(d[0]) for d in data if "face" in d]

# print(f"Faces identificadas: {sum(count_detected)}")
# print(f"Faces corretas detectadas: {sum(count_face)}")
# print(f"Faces incorretas detectadas: {sum(count_detected_with_out_1) - sum(count_face)}")
# print(f"Taxa de acertos: {sum(count_face)/sum(count_detected_with_out_1):.2f}%")
# print(f"Taxa de erros: {(sum(count_detected_with_out_1) - sum(count_face))/sum(count_detected):.2f}%")
# print(f"Clusters com apenas uma indentificação: {sum(count_detected) - sum(count_detected_with_out_1)}" )
# data = [d.split(";")[1] for d in data if ";" in d]
# print(Counter(data))

# Dados fornecidos
data = """
Ana julia ribeiro barros_1.jpg;0;
Ana klara perreira_31.jpg 0;
Ana klara perreira_35.jpg 0;
Ana laura bastos rodrigues_3.jpg 1;
Ana laura bastos rodrigues_4.jpg 1;
Ana laura bastos rodrigues_46.jpg 0;
Carlos eduardo silva coutinho_5.jpg 1;
Eduardo bretta de oliveira_37.jpg 0;
Eduardo bretta de oliveira_42.jpg 0;
Eduardo bretta de oliveira_6.jpg 1;
Eduardo bretta de oliveira_7.jpg 1;
Emilly vitoria perreira agostinho_11.jpg 1;
Emilly vitoria perreira agostinho_12.jpg 0;
Emilly vitoria perreira agostinho_9.jpg 0;
Giovanna yukari de jesus furuko_17.jpg 1;
Giovanna yukari de jesus furuko_26.jpg 0;
Isabela victoria gomes da silva_19.jpg 1;
Isabela victoria gomes da silva_20.jpg 1;
Isack braga de morais crispim_21.jpg 1;
Ivan jonathan silva cardoso_13.jpg 0;
Ivan jonathan silva cardoso_22.jpg 1;
Joao nunes dos santos netto_23.jpg 0;
Joao nunes dos santos netto_25.jpg 1;
Joao nunes dos santos netto_27.jpg 1;
Joao nunes dos santos netto_54.jpg 0;
Joao nunes dos santos netto_8.jpg 0;
Julia de bessa koch_2.jpg 1;
Julia de bessa koch_29.jpg 0;
Julia de bessa koch_32.jpg 0;
Julia de bessa koch_33.jpg 0;
Luanna paim vilaca de melo_34.jpg 1;
Luis henrique dos santos costa_38.jpg 0;
Manuella de fatima patricio da_45.jpg 1;
Manuella de fatima patricio da_50.jpg 1;
Maria eduarda zica dos santos_44.jpg 0;
Murilo de souza brito_41.jpg 1;
Pedro enzo rosa teixeira_52.jpg 0;
Rhadassa ketryn cardoso novaes_15.jpg 0;
Rhadassa ketryn cardoso novaes_16.jpg 0;
Rhadassa ketryn cardoso novaes_18.jpg 1;
Rhadassa ketryn cardoso novaes_24.jpg 0;
Rhadassa ketryn cardoso novaes_28.jpg 1;
Rhadassa ketryn cardoso novaes_30.jpg 1;
Rhadassa ketryn cardoso novaes_39.jpg 0;
Rhadassa ketryn cardoso novaes_43.jpg 1;
Rhadassa ketryn cardoso novaes_47.jpg 1;
Rhadassa ketryn cardoso novaes_48.jpg 0;
Rodrigo silverio de abreu junior_51.jpg 0;
Samuel rocha da silva_14.jpg 0;
Samuel rocha da silva_36.jpg 1;
Samuel rocha da silva_49.jpg 1;
Samuel rocha da silva_55.jpg 0;
Sarah victoria feitosa dorado_40.jpg 0;
Victor gomes pereira_53.jpg 1;
Victor hugo goncalves de faria_56.jpg 1;
Victor veras batist_10.jpg 0;
"""

# Processar os dados
lines = [line.strip() for line in data.split("\n") if line.strip()]
results = [1 if line.endswith("1;") else 0 for line in lines]

# Cálculos
total = len(results)
acertos = results.count(1)
erros = results.count(0)
assertividade = (acertos / total) * 100

print(total, acertos, erros, assertividade)