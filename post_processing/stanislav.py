
import pandas as pd
#Классы
class Human():
  df = None # датафрейм
  def __init__(self):   
    self.z = []  # список по кадрам зона входа или выхода (1 или 2) 
    self.m = 0   # результат - вошел или вышел
    
# КАСКИ ЖИЛЕТЫ

#Функции учета касок и жилетов на людях
def HelmUniform(tracker_data,pl):

  #Функции сооответствия бокса каски или жилета боксу человека
  def PartKU(h,k): # h - человек k - каска
    l = max([ h[0] ,k[0] ])
    t = max([ h[1] ,k[1] ])
    r = min([ h[2] ,k[2] ])
    b = min([ h[3] ,k[3] ])
    area_k = (k[2]-k[0])*(k[3]-k[1])
    area_i =  (r-l)*(b-t)
    if  area_i<0:
      area_i = 0
    return area_i/area_k

  dfl_k = SplitDf(tracker_data,1) # список датафреймов касок
  dfl_u = SplitDf(tracker_data,2) # список датафреймов жилетов
  #print(r_in,r_out )
  #-------------------------
  # ищем соответствие бокса каски боксу человека

  k_counter = 0 ; u_counter = 0 


  for i in range(len(pl)): # список объектов - люди
    kf = 0; uf = 0 

    for row in pl[i].df.itertuples():

      r = row.width+row.left
      b = row.height+row.top
      h_box = [row.left,row.top,r,b]

      # каски
      if len(dfl_k) != 0: 
        for q in range(len(dfl_k)):
          mask = dfl_k[q]['frame'] == row.frame

          for ro in dfl_k[q][mask].itertuples():
            pass
          
            rk = ro.width+ro.left
            bk = ro.height+ro.top
            k_box = [ro.left,ro.top,rk,bk]

            pk = PartKU(h_box,k_box) # вероятность принадлежности бокса каски к человеку
            if pk > 0.5:
              kf+=1


      # жилеты
      if len(dfl_u) != 0: 
        for q in range(len(dfl_u)):
          mask = dfl_u[q]['frame'] == row.frame
          for ro in dfl_u[q][mask].itertuples():
            ru = ro.width+ro.left
            bu = ro.height+ro.top
            u_box = [ro.left,ro.top,ru,bu]
            pk = PartKU(h_box,u_box) # вероятность принадлежности бокса каски к человеку
            if pk > 0.5:
              uf+=1

    if kf != 0 :
      k_counter+=1
    if uf != 0 :
      u_counter+=1

  return k_counter,u_counter

# Разбиение датафрейма на id
def SplitDf(tracker_data,cl):

  cls = tracker_data[tracker_data['cl'] == cl] 
  df_l = []   # список датафреймов людей   

  if len(cls) != 0:
    for i in range(cls['id'].min(),cls['id'].max()+1): # от min id до max id
      if len(cls[cls['id'] == i]) != 0: # выбор существующих в ролике id 
        tmp_h = cls[cls['id'] == i]     # выборка датафрейма с конкретным id человека

        begin = 1
        for q in range(1,len(tmp_h)):
          if q == (len(tmp_h)-1):
            df_l.append(tmp_h[begin:(q-1)])  # создание датафрейма от начала до конца датафрейма (или неразорванный датафрейм или последний в разорванном)
          if (tmp_h.iloc[q].frame-tmp_h.iloc[q-1].frame) >20: # если разрыв кадров в одном датафрейме больше 20 -  то это разные люди (20 - эмпирическое число)
            df_l.append(tmp_h[begin:(q-1)]) # создание датафрейма от начала до места разрыва 
            begin = q # bp
    

  return df_l

# Функции счетчика прошедших через турникет людей 

def Human_f(Y,tracker_data):

  #---------------------------------------------
  r_in ,  r_out = 0,0               # переменные для подсчета входящих и выходящих
  humanIdList = []  
                                    # список id людей в ролике
  df_l = SplitDf(tracker_data,0)
 
  for i in range(len(df_l)):

    h = Human();y_b_0 = None;               # создание объекта класса   h.id = i;
    h.df = df_l[i] 
    for q in range(len(df_l[i])):  
      y_b = df_l[i].iloc[q].top + df_l[i].iloc[q].height   #  (y) нижнего края бокса
      
      #----  определение стартовой позиции бокса  ---------------
      if y_b_0 == None:

        y_b_0 = y_b 
        if y_b < Y: # Y - y турникета
          h.z.append(1)  # 1 = зона входа
        if y_b > Y: 
          h.z.append(2)  # 2 = зона выхода

      #----- определение остальных позиций бокса (в остальных кадрах)-----

      if (y_b < Y)&(h.z[-1] == 2):
        h.z.append(1)     # 1 = зона входа

      if (y_b > Y)&(h.z[-1] == 1):
        h.z.append(2)     # 2 = зона выхода

    humanIdList.append(h)

  #-----------------------------------------------------------
  for j in humanIdList:       # определение наличия прохода через турникет и их подсчет

    if len(j.z) >1:
      if j.z[0] > j.z[-1]:    # старторая зона j.z[0] больше последней зоны j.z[-1] - 2>1 - это выход
        j.m = 2               # фиксация выхода для id
        r_out+=1

      if j.z[0] < j.z[-1]:    # старторая зона j.z[0] меньше последней зоны j.z[-1] - 1<2 - это вход
        j.m = 1               # фиксация входа для id
        r_in+=1  

  #-- это для касок и жилетов

  passHumList = [] # отфильтрованные объекты класса Human , прошедшие турникет

  for j in humanIdList:               # определение наличия прохода через турникет и их подсчет
    if len(j.z) >1:
      if j.z[0] > j.z[-1]:            # стартовая зона j.z[0] больше последней зоны j.z[-1] - 2>1 - это выход
        j.m = 2                       # фиксация выхода для id
        passHumList.append(j)

      if j.z[0] < j.z[-1]:            # стартовая зона j.z[0] меньше последней зоны j.z[-1] - 1<2 - это вход
        j.m = 1                       # фиксация входа для id
        passHumList.append(j)


  return  r_in,r_out , passHumList    # колическтво входов и выходов, список id прошедщих турникет

def END(p,f,turniket_dict): # работа с файлом полученного трекером

  path ='/content/drive/MyDrive/yolov8_tracking/runs/'+p+str(f)+'.txt' # путь к конктетному файлу .txt выхода трекера по конкретному видео
 
  # конвертация файла в датафрейм
  try: # если есть такой файл то обрабатываем
    # создаем датафрейм для переноса данных из .txt файла
    tracker_data = pd.read_csv(path,  sep=' ', header=None, usecols=[0,1,2,3,4,5,6] )
    tracker_data.columns = ['frame', 'id',   'left', 'top', 'width', 'height', 'cl'] 
    
    # подсчет входов и выходов  (людей)
    
    r_in,r_out, pl =  Human_f(turniket_dict[f],tracker_data)
    k,u = HelmUniform(tracker_data,pl)
    #k,u = 0,0

    dl = [f,r_in,r_out,k,u]
    return dl
  except: # если .txt файла нет (в пустых видео) , тогда нули.
    #print('empty')
    dl = [f,0,0,0,0]
    return dl

# Y турникета в группе файлов
# 470 580 850 430 510 910
'''
f1 = [1,2,3,5,8,9,10,11,12,13,15,16,17,18,19]
f2 = [6,7,14]
f3 = [20]
f4 = [21,22,23,24,31, 36,38,39,40, 42 ] #32,33,34,35,37,41,43  #430
f5 = [25,26,27,28, 29,30]
f6 = [4] # 910
'''
turniket_dict =  {1:470,2:470,3:470,4:910,5:470,\
                  6:580,7:580,8:470,9:470,10:470,\
                  11:470,12:470,13:470,14:580,15:470,\
                  16:470,17:470,18:470,19:470,20:850,\
                  21:430,22:430,23:430,24:430,25:510,\
                  26:510,27:510,28:510,29:510,30:510,\
                  31:430,32:0,33:0,34:0,35:0,\
                  36:430,37:0,38:430,39:430,40:430,\
                  41:0,42:430,43:0\
                }

# Точка входа (Запуск)
df_1 = pd.DataFrame(columns=['files','in','out','каски','жилеты']) # датафрейм для вывода данных
tracker = 'bytetrack/'
p = 't_'+tracker
for i in range(1,44):
  dl = END(p,i,turniket_dict)
  df_1.loc[i] = dl
df_1 = df_1.sort_values(by=['files'])

df_1

