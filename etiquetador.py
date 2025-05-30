# -*- coding: utf-8 -*-
"""
Created on Fri May 30 19:21:43 2025

@author: Pablo D
"""
import pandas as pd

IP_ATACANTE = "192.168.1.66"
IP_VICTIMA = "192.168.2.105"

df = pd.read_csv("ataque_udp_flood_30may_eth0_flows_avanzado.csv")

df['label'] = 'normal'
df['attack_type'] = ''

# Etiquetar como malicioso TODOS los flows UDP del atacante a la víctima (aunque tengan pocos paquetes)
df.loc[
    (df['proto'] == 17) &
    (df['src_ip'] == IP_ATACANTE) &
    (df['dst_ip'] == IP_VICTIMA),
    ['label', 'attack_type']
] = ['malicioso', 'udp_flood']

df.to_csv("ataque_udp_flood_30may_eth0_flows_etiquetado.csv", index=False)
print(df['label'].value_counts())
print(df[['src_ip','dst_ip','proto','num_packets','label','attack_type']].head(20))
