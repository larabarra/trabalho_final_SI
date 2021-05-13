# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
def valor_z(media,desvio,x,n):
    n = n**(1/2)
    z = (x - media)/(desvio/n)
    return z
def inferior(x, z,o,n):
    a = o/n**(1/2)
    i = x - z * a 
    return i
def superior(x, z,o,n):
    i = x + (z*(o/n**(1/2)))
    return i
def z_populacao (x, n, po):
    num = x - n*po
    den = n*po*(1-po)
    z = num/den**(1/2)
    return z
def media_pontual(valores):
    n = len(valores)
    total = 0;
    for i in valores:
        total = total + i
    total = total/n
    
    return total,n
    
def desvio_padrao(valores):
    media,n = media_pontual(valores)
    variancia = 0
    for i in valores:
        variancia = variancia + (i - media)**2
        
    variancia = variancia/(n-1)
    
    return variancia**(1/2)

valores = [23.01, 22.22, 22.04, 22.62, 22.59]
media,n = media_pontual(valores)
desvio = desvio_padrao(valores) 
a = valor_z(60000,3645.94,60139.7,16)
s = superior(60139.7,1.96,3645.94,16)
i = inferior(60139.7,1.96,3645.94,16)
p = z_populacao(16,200,0.1)
