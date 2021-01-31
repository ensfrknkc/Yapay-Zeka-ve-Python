import numpy as np
import pandas as pd
def basariHesapla(giris,cikis):
    from sklearn.linear_model import Perceptron 
    #from sklearn.neighbors import KNeighborsClassifier as KNN
    from sklearn.model_selection import KFold
    parcalayici = KFold(n_splits=5,shuffle=(True)) 
    ort=0
    for egitim_index, test_index in parcalayici.split(giris):
        egitim_x = giris[egitim_index,:]
        test_x = giris[test_index,:]
        egitim_y = cikis[egitim_index]
        test_y = cikis[test_index]
        #model = KNN()
        model = Perceptron() 
        model.fit(egitim_x, egitim_y)
        sonuc=model.score(test_x, test_y) 
        ort+= sonuc
    ort=ort/5 
    return ort
def simulatedAnnealing(giris,cikis,veri):
    gecicibasari=basariHesapla(giris,cikis)
    print(f"Ham Başarı:{gecicibasari}")
    basari=0
    cozum=0
    for i in range (0,50):
        gecicicozum = np.random.random(58)>0.5
        indices=np.where(gecicicozum)
        gecicigiris = veri.iloc[:,indices[0]].values
        gecicibasari=basariHesapla(gecicigiris,cikis)
        if gecicibasari>basari:
            basari=gecicibasari
            giris=gecicigiris
            cozum=gecicicozum
    print(f"Benzetilmiş Tavlama ile kullanılan Sütunlar:{cozum}")
    print(f"Benzetilmiş Tavlama En İyi Başarı:{basari}")
    return giris,basari
veri = pd.read_csv('projeveriseti.csv')
girisler = np.random.random(58)
girisstunlari=np.where(girisler)
giris = veri.iloc[:,girisstunlari[0]].values
cikis = veri.iloc[:,58].values
giris,yenibasari=simulatedAnnealing(giris,cikis,veri) 








