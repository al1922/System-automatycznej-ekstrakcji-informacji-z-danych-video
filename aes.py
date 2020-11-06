import os
import av
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pywaffle import Waffle
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.patches as mpatches
import librosa
import librosa.display
from IPython.display import Audio , display
import scipy
import xml.etree.cElementTree as Etree

def Variance(x): 
    return np.sum((x.flatten()-(np.sum(x)/x.size))**2)/x.size

def Norma(x):
    minimum = np.min(x)
    maximum = np.max(x)
    return np.asarray([(x[i]-minimum)/(maximum-minimum) for i in range(len(x))])

def NameAndFormat(path):    
    if '\\' in path :
        num = 0
        for x,i in enumerate(reversed(path)):
            if i is '\\':
                num = x
                break
        file_name = path[len(path)-x:len(path)]
        return file_name.split(os.extsep)
    else:
        return path.split(os.extsep)

def NewFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def ListCuts(lista, ogr):  
    data = [[lista[i],i] for i in range(len(lista)) if lista[i] >= ogr]
    data.append([0,0])
    cut = []
    one = 0
    for i in range(len(data)-1):
        if data[i+1][1]-data[i][1] != 1:
            try:
                cut.append(max(data[one:i], key=lambda x:x[0]))
                one = i+1
            except:
                pass
    if cut[0][1] <=  24:
        cut.pop(0)
    cut.append([lista[len(lista)-1],len(lista)])
    cut.insert(0,[lista[0],0])
    return np.asarray(cut)

def BestElements(lista,t):
    temp = [ itr if abs(lista[i,1]-lista[i+1,1]) <= t else 0  for itr ,i in enumerate(range(len(lista)-1)) ]
    temp.append(0)
    best_cut = []
    flaga = 0 
    for i in range(len(lista)):
        if temp[i] != 0 :
            if flaga == 0 :
                mini = temp[i]
            flaga = 1 
        else:
            if flaga == 1:
                maxi = temp[i-1]+1
                best_cut.append(max(lista[mini:maxi+1], key=lambda x:x[0]))
                flaga = 0 
            else:
                best_cut.append(lista[i])
    return np.asarray(best_cut)[:,1]

def SizeTest(x,y,w,h,W,H):
    xe = x+w
    ye = y+h
    if x < 0 : x = 0
    if y < 0 : y = 0
    if xe > W : xe = W
    if ye > H : ye = H
    return y, ye, x, xe 

class Open(object):
    
    def __init__(self, path):
        
        self.path = path
        self.directory = os.path.dirname(os.path.realpath(path))
        self.file = av.open(path) 
        self.name, self.format = NameAndFormat(path)
        self.frames = self.file.streams.video[0].frames
        self.fps =  round(self.file.streams.video[0].framerate)
        self.duration = self.frames/self.fps
        self.height = self.file.streams.video[0].height
        self.width = self.file.streams.video[0].width
        self.pixel_format = self.file.streams.video[0].pix_fmt
        
        self.lightintensity = None
        self.colordomination = None
        self.dynamic = None
        self.extractobject = None
        self.shot = None
        self.vocal = None
        self.chroma = None
        
    def ownedit(self,start ,end, save_format=None, save_path=None, name=None):
        if save_format == None:
            save_format = self.format
        if name == None:
            name = self.name+'_edit'
        if save_path == None:
            save_path = '"'+str(self.directory)+'\\'+str(name)+'.'+str(save_format)+'"'
        elif save_path is not None:
            save_path = '"'+str(save_path)+'\\'+str(name)+'.'+str(save_format)+'"'
            
        os.system('ffmpeg -y -i "'+str(self.path)+'" -ss '+str(start)+' -t '+str(end-start)+' -async 1 '+ str(save_path))
        
    def savedata(self, path=None, name=None):
        if path == None:
            NewFolder(str(self.directory)+'\\'+str(self.name))
            path = str(self.directory)+'\\'+str(self.name)
        if name == None:
            name = self.name
            
        root = Etree.Element("root")
        doc = Etree.SubElement(root, self.name)

        Etree.SubElement(doc, "data1", name="name").text = str(self.name)
        Etree.SubElement(doc, "data2", name="format").text = str(self.format)
        Etree.SubElement(doc, "data3", name="frames").text = str(self.frames)
        Etree.SubElement(doc, "data4", name="duration").text = str(self.duration)
        Etree.SubElement(doc, "data5", name="fps").text = str(self.fps)
        Etree.SubElement(doc, "data6", name="height").text = str(self.height)
        Etree.SubElement(doc, "data7", name="width").text = str(self.width)
        Etree.SubElement(doc, "data8", name="pixel_format").text = str(self.pixel_format)
        Etree.SubElement(doc, "data9", name="shot").text = str(self.shot)
        Etree.SubElement(doc, "data10", name="extractobject").text = str(self.extractobject)
        Etree.SubElement(doc, "data11", name="colordomination").text = str(self.colordomination)
        Etree.SubElement(doc, "data12", name="lightintensity").text = str(self.lightintensity)
        Etree.SubElement(doc, "data13", name="dynamic").text = str(self.dynamic)
        Etree.SubElement(doc, "data14", name="vocal").text = str(self.vocal)
        Etree.SubElement(doc, "data15", name="chroma").text = str(self.chroma)

        tree = Etree.ElementTree(root)
        tree.write(str(path)+'\\'+str(name)+".xml")

class Shot(): 
    
    def __init__(self, video):
        self.video = video 
        self.file = av.open(self.video.path) 
        self.arraymean()
        self.parameters()
    
    def arraymean(self):
        video_array = np.zeros((self.video.frames,3)) 
        for iteration,frame in enumerate(self.file.decode(video=0)):
            video_array[iteration] = cv2.mean(frame.to_ndarray(format='bgr24'))[:3]
        self.avg_colors = video_array                

    def parameters(self, tolerance=0.03, distance=48):
        colors = np.sum(self.avg_colors**2,axis=1)
        wyn = BestElements(ListCuts(Norma([Variance(colors[i:i+4]) for i in range(np.size(colors,axis=0))]),tolerance),distance)
        self.secends = np.round(wyn/self.video.fps,3)
        self.frames = wyn
        self.video.shot = {'secend': self.secends, 'frame': self.frames}
        
    def saveshot(self,path=None ,name=None ,save_format=None,folder= False):
        if name == None:
            name = str(self.video.name)+'_shot'
        if save_format == None:
            save_format = self.video.format
        if path == None:
            path = self.video.directory
            NewFolder(str(path)+'\\'+str(self.video.name))
            NewFolder(str(path)+'\\'+str(self.video.name)+'\\Shots')
            for itr in range(self.secends.size-1):
                os.system('ffmpeg -y -i "'+str(self.video.path)+'" -ss '+str(self.secends[itr])+' -t '+str(self.secends[itr+1]-self.secends[itr])+' -async 1 "'+str(path)+'\\'+str(self.video.name)+'\\Shots\\'+str(name)+str(itr)+'.'+str(save_format)+'"')
        
        elif path is not None:
            if folder == False:
                for itr in range(self.secends.size-1):
                    os.system('ffmpeg -y -i "'+str(self.video.path)+'" -ss '+str(self.secends[itr])+' -t '+str(self.secends[itr+1]-self.secends[itr])+' -async 1 "'+str(path)+'\\'+str(name)+str(itr)+'.'+str(save_format)+'"')
            elif folder == True:
                NewFolder(str(path)+'\\'+str(self.video.name))
                for itr in range(self.secends.size-1):
                    os.system('ffmpeg -y -i "'+str(self.video.path)+'" -ss '+str(self.secends[itr])+' -t '+str(self.secends[itr+1]-self.secends[itr])+' -async 1 "'+str(path)+'\\'+str(self.video.name)+'\\'+str(name)+str(itr)+'.'+str(save_format)+'"')

class ExtractObject():
    def __init__(self, video, probability=0.7, frequency=48, save_picture=True, save_path=None ,save_format='png'):
        self.video = video
        self.file = av.open(self.video.path) 
        self.dir_recognizon = os.path.dirname(os.path.realpath('aes'))
        self.probability = probability
        self.frequency = frequency
        self.save_format = save_format
        self.save_picture = save_picture
        if self.save_picture == True:
            if save_path == None:
                NewFolder(str(self.video.directory)+'\\'+str(self.video.name))
                NewFolder(str(self.video.directory)+'\\'+str(self.video.name)+'\\ObjectDetection')
                self.path = str(self.video.directory)+'\\'+str(self.video.name)+'\\ObjectDetection'
            elif save_path is not None :
                self.path = save_path
            if save_format == None:
                self.save_format = save_format

        self.extract()
        
    def extract(self):
        
        name_class = [itr.rstrip() for itr in open(str(self.dir_recognizon)+'\\recognition\\coco.names')]
        net = cv2.dnn.readNetFromDarknet(str(self.dir_recognizon)+'\\recognition\\yolov3.cfg',str(self.dir_recognizon)+'\\recognition\\yolov3.weights')
        layer = [net.getLayerNames()[itr[0]-1] for itr in net.getUnconnectedOutLayers()]
        name_use = np.zeros(len(name_class))
        
        skip = 0 
        for frame in self.file.decode(video=0):
            if skip <= self.frequency:
                skip+=1
            else:
                skip = 0
                img = frame.to_ndarray(format='bgr24')
                net.setInput(cv2.dnn.blobFromImage(img,1/255.0, (416, 416),swapRB=True, crop=False))
                layerOutputs = net.forward(layer)

                boxes = []
                confidences = []
                names = []

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        name = np.argmax(scores)
                        confidence = scores[name]

                        if confidence > self.probability:
                            (x, y, w, h) = (detection[0:4] * np.array([self.video.width, self.video.height, self.video.width, self.video.height])).astype("int")
                            boxes.append([int(x - (w / 2)), int(y - (h / 2)), int(w), int(h)])
                            confidences.append(float(confidence))
                            names.append(name)

                indexs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
                if len(indexs) > 0:
                    for i in indexs.flatten():
                        x,y,w,h = boxes[i]
                        yl,yr,xl,xr =SizeTest(x,y,w,h,self.video.width,self.video.height)
                        if self.save_picture == True:
                            cv2.imwrite(str(self.path)+'\\'+str(name_class[names[i]])+str(int(name_use[names[i]]))+'.'+str(self.save_format),img[yl:yr,xl:xr])
                        name_use[names[i]] += 1
                        
        self.video.extractobject = {'name':name_class, 'use':name_use}    

class ColorDomination():
    def __init__(self,video):
        self.video = video
        self.file = av.open(self.video.path) 
        self.colorsextract()
        self.partition()
        self.showplot(close=True)
        
    def colorsextract(self):
        y = int(0.3*self.video.height)
        x = int(0.2*self.video.width)
        clas = KMeans(5,max_iter=50)
        skip = 0
        colors = []
        for frame in self.file.decode(video=0):
            if skip < 24:
                skip +=1
            else:
                skip = 0
                img = cv2.resize(frame.to_ndarray(format='bgr24')[y:self.video.height-y,x:self.video.width-x],(128,128))
                temp = img.reshape((img.shape[0] * img.shape[1], 3))
                clas.fit(temp)
                colors.append(clas.cluster_centers_)
        self.simple_colors = np.asarray(colors)
         
    def partition(self,quantity = 10):
        self.quantity = quantity
        colors = self.simple_colors.reshape((self.simple_colors.shape[0] * self.simple_colors.shape[1], 3))
        clas = KMeans(self.quantity,max_iter=6000)
        clas.fit(colors)
        self.colors = [[i[2],i[1],i[0]] for i in (clas.cluster_centers_/255.0) ]

        unique, counts = np.unique(clas.labels_, return_counts=True)
        data = dict(zip(unique, counts))
        for i in range(len(data)): data[i] = np.round((data[i]/clas.labels_.size)*200,2)
        self.occurrence = data
        self.video.colordomination = {'colors': self.colors ,'occurrence': self.occurrence}
    
        
    def showplot(self,rows=6, cor=0, distance=0.3, figsize=(17,10), close=False):
        fig = plt.figure(
            FigureClass=Waffle,
            figsize=figsize,
            rows=rows, 
            values=self.occurrence, 
            colors=(self.colors),
            title={'label': '# Colors domination' , 'loc': 'center','fontsize': 14},
            labels=["{0} ({1}%)".format(k, v) for k, v in self.occurrence.items()],
            legend={'loc': 'lower left', 'bbox_to_anchor': (0, -distance), 'ncol': len(self.occurrence)-cor, 'framealpha': 0},
            starting_location='NW',
            
            )
        fig.set_facecolor([0.9,0.9,0.9])
        self.figure = fig
        if close is True :
            plt.close()
        else:
            plt.show()
        
    def saveplot(self,path=None, name=None, save_format='png'):

        if name == None:
            name = str(self.video.name)+'_colors_domination'
        if path == None:
            path = self.video.directory
            NewFolder(str(path)+'\\'+str(self.video.name))            
            self.figure.savefig(str(path)+'\\'+str(self.video.name)+'\\'+str(name)+'.'+str(save_format))
        elif path is not None:
            self.figure.savefig(str(path)+'\\'+str(name)+'.'+str(save_format))

class LightIntensity():
    def __init__(self,video):
        self.video = video
        self.file = av.open(self.video.path)
        self.light()
        self.showplot(clear=True)
        
    def light(self):
        y = int(0.2*self.video.height)
        x = int(0.1*self.video.width)
        light = np.zeros((self.video.frames))
        for itr, frame in enumerate(self.file.decode(video=0)):
            light[itr] = cv2.mean((cv2.cvtColor(frame.to_ndarray(format='bgr24')[y:self.video.height-y,x:self.video.width-x], cv2.COLOR_BGR2GRAY)))[0]
            
        self.light = light    
        n, bins, patches = plt.hist(light,35,density=1)
        plt.close()
        wyn = sns.kdeplot(np.array(light), bw=1,legend =True).get_lines()[0].get_data()
        plt.close()
        self.density = [np.max(wyn[1]),wyn[0][np.argmax(wyn[1])]]
        self.average = np.sum(light)/len(light)
        self.video.lightintensity = {'frames':self.light, 'bins':bins, 'bins_value':n , 'density':self.density, 'average':self.average}
        
    def showplot(self,clear=False):
        fig, ax = plt.subplots(figsize=(15,8))
        fig.set_facecolor([0.99,0.99,0.99])
        cm = plt.cm.get_cmap('inferno')
        n, bins, patches = ax.hist(self.light,35,density=1)
        col = (n-n.min())/(n.max()-n.min())
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        avg = mpatches.Patch( color='orange', label='Average: '+str(np.round(self.average,2)))
        den = mpatches.Patch( color='coral', label='Density: '+str(np.round(self.density[1],2)))
        ax.legend(handles=[avg,den])
        ax.set_xlabel('Value (0, 255)',fontsize=12)
        ax.set_ylabel('Quantity',fontsize=12)
        ax.set_title('# Light intensity',fontsize=16)
        sns.kdeplot(np.array(self.light), bw=1, legend=True, color="coral")
        self.figure = fig
        if clear == False:
            plt.show()
        elif clear == True:
            plt.close()
        
    def saveplot(self,path=None, name=None, save_format='png'):
        if name == None:
            name = str(self.video.name)+'_light_intensity'
        if path == None:
            path = self.video.directory
            NewFolder(str(path)+'\\'+str(self.video.name))
            self.figure.savefig(str(path)+'\\'+str(self.video.name)+'\\'+str(name)+'.'+str(save_format))
        elif path is not None:
            self.figure.savefig(str(path)+'\\'+str(name)+'.'+str(save_format))

class Dynamic():
    def __init__(self,video):
        self.video = video
        self.shot = Shot(video)
        self.file = av.open(self.video.path)
        self.cuts = self.shot.frames
        self.move = np.subtract(np.zeros(self.video.frames),np.ones(self.video.frames))
        self.moving()
        self.showplot(clear=True)
        
    def moving(self):
        dziel = self.video.height * self.video.width
        self.cuts = np.append(self.cuts,[self.video.frames+1])
        part = 0 
        for itr, frame in enumerate(self.file.decode(video=0)):
            if itr >= self.cuts[part]:
                part+=1
                sub = cv2.createBackgroundSubtractorMOG2()
            self.move[itr] = np.divide(np.sum(sub.apply(frame.to_ndarray(format='bgr24'))),dziel)
            
        self.movie = np.array( [i for i in self.move if i > 1])
        n, bins, patches = plt.hist(self.movie,35,density=1)
        plt.close()
        wyn = sns.kdeplot(np.array(self.movie), bw=1,legend =True).get_lines()[0].get_data()
        plt.close()
        self.density = [np.max(wyn[1]),wyn[0][np.argmax(wyn[1])]]
        self.average = np.sum(self.movie)/len(self.movie)
        self.video.dynamic = {'frames':self.movie, 'bins':bins, 'bins_value':n , 'density':self.density, 'average':self.average}
        
    def showplot(self,clear=False):
        fig, ax = plt.subplots(figsize=(15,8))
        fig.set_facecolor([0.99,0.99,0.99])
        cm = plt.cm.get_cmap('YlGnBu')
        n, bins, patches = ax.hist(self.movie,35,density=1)
        col = (n-n.min())/(n.max()-n.min())
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        avg = mpatches.Patch( color='royalblue' ,label='Average: '+str(np.round(self.average,2)))
        den = mpatches.Patch( color='black', label='Density: '+str(np.round(self.density[1],2)))
        ax.legend(handles=[avg,den])
        ax.set_xlabel('Value (0, 255)',fontsize=12)
        ax.set_ylabel('Quantity',fontsize=12)
        ax.set_title('# Dynamism',fontsize=16)
        sns.kdeplot(np.array(self.movie), bw=1, legend =True, color="black")
        self.figure = fig
        if clear == False:
            plt.show()
        elif clear == True:
            plt.close()
        
    def saveplot(self,path=None, name=None, save_format='png'):
        if name == None:
            name = str(self.video.name)+'_dynamism'
        if path == None:
            path = self.video.directory
            NewFolder(str(path)+'\\'+str(self.video.name))  
            self.figure.savefig(str(path)+'\\'+str(self.video.name)+'\\'+str(name)+'.'+str(save_format))
        elif path is not None:
            self.figure.savefig(str(path)+'\\'+str(name)+'.'+str(save_format))

class Vocal():

    def __init__(self,video,margin_i=20, margin_v=20,power=1,ownfolder=False):
        self.video = video
        self.orginaluadio(ownfolder)
        self.extractvocal(margin_i,margin_v,power)
        self.showplot(clear=True)
        
    def orginaluadio(self,folder):
        if folder is False :
            NewFolder(self.video.directory+'\\'+self.video.name)
            self.save_path = self.video.directory+'\\'+self.video.name
            self.save_format = 'mp3'
            if os.path.isfile(str(self.save_path)+'\\'+self.video.name+'_audio.'+str(self.save_format)) is False :
                self.video.ownedit(0,self.video.duration,save_format = self.save_format ,save_path = self.save_path ,name = self.video.name+'_audio')
            self.orginal = self.video.directory+'\\'+self.video.name+'\\'+self.video.name+'_audio.'+self.save_format
            
        elif folder is True:
            NewFolder(self.video.directory+'\\Sound')
            self.save_path = self.video.directory+'\\Sound'
            self.save_format = 'mp3'
            if os.path.isfile(str(self.save_path)+'\\'+self.video.name+'_audio.'+str(self.save_format)) is False :
                self.video.ownedit(0,self.video.duration,save_format = self.save_format ,save_path = self.save_path ,name = self.video.name+'_audio')
            self.orginal =  self.save_path+'\\'+self.video.name+'_audio.'+self.save_format
            
    def extractvocal(self,margin_i,margin_v,power):
        # Code source: Brian McFee
        # License: ISC
        # Link : https://librosa.github.io/librosa/auto_examples/plot_vocal_separation.html#sphx-glr-auto-examples-plot-vocal-separation-py , 23.11.2019 13:36
        self.y, self.sr = librosa.load(self.orginal)
        S_full, phase = librosa.magphase(librosa.stft(self.y))

        S_filter = librosa.decompose.nn_filter(S_full,aggregate=np.median,metric='cosine',width=int(librosa.time_to_frames(2, sr=self.sr)))
        S_filter = np.minimum(S_full, S_filter)
        mask_i = librosa.util.softmask(S_filter,margin_i * (S_full - S_filter),power=power*4)
        mask_v = librosa.util.softmask(S_full - S_filter,margin_v * S_filter,power=power*1)
        
        self.S_foreground = mask_v * S_full
        self.S_background = mask_i * S_full
        self.S_full = S_full
        self.foreground = librosa.istft(self.S_foreground)
        self.background = librosa.istft(self.S_background)
        self.video.vocal = {'orginal':self.y,'rate':self.sr, 'foreground':self.S_foreground, 'background':self.S_background}
        
    def showplot(self,time=20,clear=False):
        if time is not None:
            x = int((self.video.duration/2)-int(time/2))
            z = int((self.video.duration/2)+int(time/2))
            idx = slice(*librosa.time_to_frames([x,z], sr=self.sr))
        
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(self.S_full[:, idx], ref=np.max),
                                 y_axis='log', sr=self.sr)
        plt.title('# Full spectrum - '+str(time)+' middle frames')
        plt.colorbar()

        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(self.S_background[:, idx], ref=np.max),
                                 y_axis='log', sr=self.sr)
        plt.title('# Background - '+str(time)+' middle frames')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(self.S_foreground[:, idx], ref=np.max),
                                 y_axis='log', x_axis='time', sr=self.sr)
        plt.title('# Foreground - '+str(time)+' middle frames')
        plt.colorbar()
        plt.tight_layout()
        self.figure = fig
        if clear == False:
            plt.show()
        elif clear == True:
            plt.close()
    
    def showsound(self):
        print('Orginal')
        display(Audio(data=self.y, rate =self.sr))
        print('Foreground')
        display(Audio(data=self.foreground, rate = self.sr))
        print('Background')
        display(Audio(data=self.background, rate =self.sr))
        
    def saveplot(self,path=None, name=None, save_format='png'):
        if name == None:
            name = self.video.name+'_vocal'
        if path == None:
            path = self.save_path
            NewFolder(self.save_path)
        elif path is not None:
            path = path
            
        self.figure.savefig(str(path)+'\\'+str(name)+'.'+str(save_format))                    

    def savebackground(self,path=None, name=None, save_format=None):
        if save_format == None:
            save_format = self.save_format
        if name == None:
            name = self.video.name+'_background'
        if path == None:
            path = self.save_path
        elif path is not None:
            path = path
        librosa.output.write_wav(str(path)+'\\'+str(name)+'.'+str(save_format), self.background, self.sr)
        
    def saveforeground(self,path=None, name=None, save_format=None):
        if save_format == None:
            save_format = self.save_format
        if name == None:
            name = self.video.name+'_foreground'
        if path == None:
            path = self.save_path
        elif path is not None:
            path = path
        librosa.output.write_wav(str(path)+'\\'+str(name)+'.'+str(save_format), self.foreground, self.sr)

class Chroma():
    
    def __init__(self,video,ownfolder=False):
        self.video = video
        self.orginaluadio(ownfolder)
        self.extractchroma()
        self.showplot(clear=True)
        
    def orginaluadio(self,folder):
        if folder is False :
            NewFolder(self.video.directory+'\\'+self.video.name)
            self.save_path = self.video.directory+'\\'+self.video.name
            self.save_format = 'mp3'
            if os.path.isfile(str(self.save_path)+'\\'+self.video.name+'_audio.'+str(self.save_format)) is False :
                self.video.ownedit(0,self.video.duration,save_format = self.save_format ,save_path = self.save_path ,name = self.video.name+'_audio')
            self.orginal = self.video.directory+'\\'+self.video.name+'\\'+self.video.name+'_audio.'+self.save_format
            
        elif folder is True:
            NewFolder(self.video.directory+'\\Sound')
            self.save_path = self.video.directory+'\\Sound'
            self.save_format = 'mp3'
            if os.path.isfile(str(self.save_path)+'\\'+self.video.name+'_audio.'+str(self.save_format)) is False :
                self.video.ownedit(0,self.video.duration,save_format = self.save_format ,save_path = self.save_path ,name = self.video.name+'_audio')
            self.orginal =  self.save_path+'\\'+self.video.name+'_audio.'+self.save_format
        
    def extractchroma(self):
        # Code source: Brian McFee
        # License: ISC
        # https://librosa.github.io/librosa/auto_examples/plot_chroma.html#sphx-glr-auto-examples-plot-chroma-py ,23.11.2019, 21:51 
        self.y, self.sr = librosa.load(self.orginal)
        self.chroma_orig = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        y_harm = librosa.effects.harmonic(y=self.y, margin=8)
        chroma_os_harm = librosa.feature.chroma_cqt(y=y_harm, sr=self.sr, bins_per_octave=12*3)
        chroma_filter = np.minimum(chroma_os_harm,librosa.decompose.nn_filter(chroma_os_harm,aggregate=np.median,metric='cosine'))
        self.chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
        self.processed = librosa.istft(self.chroma_smooth)
        self.video.chroma ={'orginal_chroma':self.chroma_orig, 'processed':self.chroma_smooth}
        
    def showplot(self,time=20,clear=False):
        if time is not None:
            x = int((self.video.duration/2)-int(time/2))
            z = int((self.video.duration/2)+int(time/2))
            idx = tuple([slice(None), slice(*list(librosa.time_to_frames([x, z])))])
        
        fig = plt.figure(figsize=(11, 6))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(self.chroma_orig[idx], y_axis='chroma')
        plt.colorbar()
        plt.ylabel('Original')
        plt.subplot(2, 1, 2)
        librosa.display.specshow(self.chroma_smooth[idx], y_axis='chroma', x_axis='time')
        plt.ylabel('Processed')
        plt.colorbar()
        plt.tight_layout()
        self.figure = fig
        if clear == False:
            plt.show()
        elif clear == True:
            plt.close()
            
    def saveplot(self,path=None, name=None, save_format='png'):
        if name == None:
            name = self.video.name+'_chroma'
        if path == None:
            path = self.save_path
            NewFolder(self.save_path)
        elif path is not None:
            path = path
            
        self.figure.savefig(str(path)+'\\'+str(name)+'.'+str(save_format))          

class ExtractDataForAll():
    def __init__(self,movies,tool,parameters=None):
        self.shots = movies
        self.tool = tool
        self.name = tool
        self.parameters = parameters

    def execute(self):
        tolerance=0.03
        distance=48
        if self.name == 'Shot':
            if self.parameters is not None:
                for i in self.parameters:
                    if i == 'tolerance':tolerance = self.parameters['tolerance']
                    if i == 'distance':distance = self.parameters['distance']   

                        
            self.tool = eval(self.tool)
            self.lista = []
            for shot in range(len(self.shots.files)):            
                self.lista.append(self.tool(self.shots.files[shot]))
                self.lista[shot].parameters(distance=distance,tolerance=tolerance)
                
            self.shots.shot = [self.shots.files[i].shot for i in range(len(self.shots.name))]   
            
        if self.name == 'ExtractObject':
            save_path = str(self.shots.path)+'\\Objects\\'
            save_format='png'
            frequency=48
            probability=0.7
            if self.parameters is not None:
                for i in self.parameters:
                    if i == 'probability':probability = self.parameters['probability']
                    if i == 'frequency':frequency = self.parameters['frequency']
                    if i == 'path':save_path = self.parameters['path']
                    if i == 'save_format':save_format = self.parameters['save_format']

            NewFolder(str(self.shots.path)+'\\Objects')
            self.tool = eval(self.tool)
            self.lista = []
            for shot in range(len(self.shots.files)):
                NewFolder(str(self.shots.path)+'\\Objects\\'+str(self.shots.files[shot].name))
                self.lista.append(self.tool(self.shots.files[shot],save_path=save_path+str(self.shots.files[shot].name),save_format=save_format,frequency=frequency,probability=probability))
                
            self.shots.extractobject = [self.shots.files[i].extractobject for i in range(len(self.shots.name))]  
    
        if self.name == 'ColorDomination':
            quantity=5 
            rows=6
            cor=0
            distance=0.3
            figsize=(17,10)
            if self.parameters is not None:
                for i in self.parameters:
                    if i == 'quantity':quantity = self.parameters['quantity']
                    if i == 'rows':rows = self.parameters['rows']   
                    if i == 'cor':cor = self.parameters['cor']   
                    if i == 'distance':distance = self.parameters['distance']   
                    if i == 'figsize':figsize = self.parameters['figsize']   
                        
            self.tool = eval(self.tool)
            self.lista = []
            for shot in range(len(self.shots.files)):
                self.lista.append(self.tool(self.shots.files[shot]))
                self.lista[shot].partition(quantity)
                self.lista[shot].showplot(rows=rows, cor=cor, distance=distance, figsize=figsize, close=True )
                
            self.shots.colordomination = [self.shots.files[i].colordomination for i in range(len(self.shots.name))]  

        if self.name == 'LightIntensity' or self.name == 'Dynamic':
            self.tool = eval(self.tool)
            self.lista = []
            for shot in range(len(self.shots.files)):
                self.lista.append(self.tool(self.shots.files[shot]))
                
            self.shots.lightintensity = [self.shots.files[i].lightintensity for i in range(len(self.shots.name))]   
            self.shots.dynamic = [self.shots.files[i].dynamic for i in range(len(self.shots.name))]
            
        if self.name == 'Chroma' :
            self.tool = eval(self.tool)
            self.lista = []
            for shot in range(len(self.shots.files)):
                self.lista.append(self.tool(self.shots.files[shot],ownfolder=True))   
            self.shots.chroma = [self.shots.files[i].chroma for i in range(len(self.shots.name))]  
                
        if self.name == 'Vocal':
            margin_i=20 
            margin_v=20
            power=1
            if self.parameters is not None:
                for i in self.parameters:
                    if i == 'margin_i':margin_i = self.parameters['margin_i']
                    if i == 'margin_v':margin_v = self.parameters['margin_v']   
                    if i == 'power':power = self.parameters['power']   
                        
            self.tool = eval(self.tool)
            self.lista = []
            for shot in range(len(self.shots.files)):
                self.lista.append(self.tool(self.shots.files[shot],ownfolder=True,margin_i=margin_i,margin_v=margin_v,power=power))
                
            self.shots.vocal = [self.shots.files[i].vocal for i in range(len(self.shots.name))]  
    
    def showplot(self,parameters=None):
        if self.name == 'Chroma':
            time = 20
            if self.parameters is not None:
                for i in self.parameters:
                    if i == 'time':time = self.parameters['time'] 
            for shot in range(len(self.shots.files)):
                self.lista[shot].showplot(time)      
                
        if self.name == 'Vocal':
            time = 20
            if self.parameters is not None:
                for i in self.parameters:
                    if i == 'time':time = self.parameters['time']
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].showplot(time)
        
        if self.name == 'LightIntensity' or self.name == 'Dynamic':
            for shot in range(len(self.shots.files)):
                self.lista[shot].showplot()
                
        if self.name == 'ColorDomination':
            rows=6
            cor=0
            distance=0.3
            figsize=(17,10)
            if parameters is not None:
                for i in parameters:
                    if i == 'quantity': quantity = parameters['quantity']
                    if i == 'rows': rows = parameters['rows']   
                    if i == 'cor': cor = parameters['cor']   
                    if i == 'distance':distance = parameters['distance']   
                    if i == 'figsize': figsize = parameters['figsize']   
            
            for shot in range(len(self.shots.files)):
                self.lista[shot].showplot(rows=rows, cor=cor, distance=distance, figsize=figsize,close=False)
    
    def saveplot(self,parameters=None):
        
        if self.name == 'Chroma':
            save_format='png'
            NewFolder(str(self.shots.path)+'\\ChromaPlot')
            save_path = str(self.shots.path)+'\\ChromaPlot\\'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].saveplot(path=save_path,save_format=save_format)
        
        if self.name == 'Vocal':
            save_format='png'
            NewFolder(str(self.shots.path)+'\\VocalPlot')
            save_path = str(self.shots.path)+'\\VocalPlot\\'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].saveplot(path=save_path,save_format=save_format)
                
        if self.name == 'Dynamic':
            save_format='png'
            NewFolder(str(self.shots.path)+'\\Dynamic')
            save_path = str(self.shots.path)+'\\Dynamic\\'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].saveplot(path=save_path,save_format=save_format)
        
        if self.name == 'LightIntensity':
            save_format='png'
            NewFolder(str(self.shots.path)+'\\LightIntensity')
            save_path = str(self.shots.path)+'\\LightIntensity\\'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].saveplot(path=save_path,save_format=save_format)
                
            
        if self.name == 'ColorDomination':
            save_format='png'
            NewFolder(str(self.shots.path)+'\\ColorsDomination')
            save_path = str(self.shots.path)+'\\ColorsDomination\\'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].saveplot(path=save_path,save_format=save_format)
    
    def showsound(self):    
        if self.name == 'Vocal':
            for shot in range(len(self.shots.files)):
                self.lista[shot].showsound()
    
    def saveforeground(self,parameters=None):
        if self.name == 'Vocal':
            save_format='mp3'
            NewFolder(str(self.shots.path)+'\\Sound\\foreground')
            save_path = str(self.shots.path)+'\\Sound\\foreground'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].saveforeground(path=save_path,save_format=save_format)

    def savebackground(self,parameters=None):
        if self.name == 'Vocal':
            save_format='mp3'
            NewFolder(str(self.shots.path)+'\\Sound\\background')
            save_path = str(self.shots.path)+'\\Sound\\background\\'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].savebackground(path=save_path,save_format=save_format)
                
    def saveshot(self,parameters=None):
        if self.name == 'Shot':
            save_format='mp4'
            NewFolder(str(self.shots.path)+'\\AllShots')
            save_path = str(self.shots.path)+'\\AllShots\\'
            if parameters is not None:
                for i in parameters:
                    if i == 'path': save_path = parameters['path']
                    if i == 'save_format': save_format = parameters['save_format']       
                        
            for shot in range(len(self.shots.files)):
                self.lista[shot].saveshot(path=save_path,save_format=save_format, folder=True)            

class OpenAll():
    def __init__(self,path):
        self.path = path
        self.lista = []
        self.findshots()
        self.files = [ Open(shot) for shot in self.lista]
        self.len_of_elements = range(len(self.files))
        
        self.name = [self.files[i].name for i in self.len_of_elements]
        self.format = [self.files[i].format for i in self.len_of_elements]
        self.frames = [self.files[i].frames for i in self.len_of_elements]
        self.fps =  [self.files[i].fps for i in self.len_of_elements]
        self.duration = [self.files[i].duration for i in self.len_of_elements]
        self.height = [self.files[i].height for i in self.len_of_elements]
        self.width = [self.files[i].width for i in self.len_of_elements]
        self.pixel_format = [self.files[i].pixel_format for i in self.len_of_elements]
        
        self.lightintensity = None
        self.colordomination = None
        self.dynamic = None
        self.extractobject = None
        self.shot = None
        self.vocal = None
        self.chroma = None
                
    def findshots(self):
        for file in os.listdir(self.path):
            try:
                _, shot_format = NameAndFormat(self.path+'\\'+str(file))
                if file.endswith('.'+str(shot_format)):
                    self.lista.append(os.path.join(self.path, file))
            except:
                pass
                
    def savedata(self,path=None, separate = True):
        if path is None:
            path = self.path
            NewFolder(str(path)+'\\AllData')

        if separate == False:

            root = Etree.Element("root")
            doc = Etree.SubElement(root, "data")

            Etree.SubElement(doc, "data1", name="name").text = str(self.name)
            Etree.SubElement(doc, "data2", name="format").text = str(self.format)
            Etree.SubElement(doc, "data3", name="frames").text = str(self.frames)
            Etree.SubElement(doc, "data4", name="duration").text = str(self.duration)
            Etree.SubElement(doc, "data5", name="fps").text = str(self.fps)
            Etree.SubElement(doc, "data6", name="height").text = str(self.height)
            Etree.SubElement(doc, "data7", name="width").text = str(self.width)
            Etree.SubElement(doc, "data8", name="pixel_format").text = str(self.pixel_format)
            Etree.SubElement(doc, "data9", name="shot").text = str(self.shot)
            Etree.SubElement(doc, "data10", name="extractobject").text = str(self.extractobject)
            Etree.SubElement(doc, "data11", name="colordomination").text = str(self.colordomination)
            Etree.SubElement(doc, "data12", name="lightintensity").text = str(self.lightintensity)
            Etree.SubElement(doc, "data13", name="dynamic").text = str(self.dynamic)
            Etree.SubElement(doc, "data14", name="vocal").text = str(self.vocal)
            Etree.SubElement(doc, "data15", name="chroma").text = str(self.chroma)

            tree = Etree.ElementTree(root)
            tree.write(str(path)+'\\AllData\\AllData.xml')
            
        elif separate == True:
            for shot in range(len(self.lista)):
                self.files[shot].savedata(path=str(path)+'\\AllData')