1. Predição

data.head(1)

  ***Exemplo:**jogador para ser atacante 2.06m e 99Kg


altura_padronizada = (2.06 - altura_media) / altura_desvio_padrao
print(altura_padronizada)

jogador = np.array([altura_padronizada, 0, 0, 1, 0, 0, 0, 0])
print(jogador)

peso = model_v2.predict(jogador.reshape(1, -1))
print(peso)


    ***Exemplo:**jogador para ser atacante 2.06m e 99Kg


altura_padronizada = (2.06 - altura_media) / altura_desvio_padrao
print(altura_padronizada)

jogador = np.array([altura_padronizada, 1, 0, 0, 0, 0, 0, 0])
print(jogador)

peso = model_v2.predict(jogador.reshape(1 , -1))
print(peso)