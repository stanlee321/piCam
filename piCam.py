#!/usr/bin/env python
import numpy as np
import cv2
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
#import cv2.cv as cv

help_message = '''
USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Lst
'''

def Find_color(imagen):
	global color
	
	hsv=cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
	blurAm=cv2.inRange(hsv,Amarillo1, Amarillo2)
	blurAma=cv2.GaussianBlur(blurAm,(5,5),0)
	r1,mascaraAma=cv2.threshold(blurAma,30,255, cv2.THRESH_BINARY)

	blurRoj=cv2.inRange(hsv,rojo1,rojo2)
	blurRojo=cv2.GaussianBlur(blurRoj,(5,5),0)
	r2,mascaraRoj=cv2.threshold(blurRojo,50,255, cv2.THRESH_BINARY)

	blurVer=cv2.inRange(hsv,verde1,verde2)
	blurVerd=cv2.GaussianBlur(blurVer,(7,7),0)
	r3,mascaraVerd=cv2.threshold(blurVerd,30,255, cv2.THRESH_BINARY)
	
	#cv2.imshow('imagennn',imagen)
	#cv2.imshow('Amarillo', mascaraAma)
	#cv2.imshow('Rojo',mascaraRoj)
	#cv2.imshow('verde',mascaraVerd)
	im, cotours,hierarchy =cv2.findContours(mascaraRoj,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im2, cotou,hierarch =cv2.findContours(mascaraVerd,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im3, cotu,hierarc =cv2.findContours(mascaraAma,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if (len(cotours)+len(cotou)+len(cotu))!=0:
		if len(cotours)!=0:
			for cantRo in cotours:
				if  cv2.contourArea(cantRo) >=5:
					print ('Rojo')
					color=1
		if len(cotou)!=0:
			for cantVer in cotou:
				if  cv2.contourArea(cantVer) >=10:
					print ('verde')
					color=0
		if len(cotu)!=0:
			for cantAma in cotu:
				if  cv2.contourArea(cantAma) >=5:
					print ('amarillo')
					color=2
	else:
		print ('No hay semaforo')
		color=3
	return color
Amarillo1=np.array([16,100,71], dtype=np.uint8)
Amarillo2=np.array([35,255,255], dtype=np.uint8)
rojo1=np.array([160,100,100], dtype=np.uint8)
rojo2=np.array([179,255,255], dtype=np.uint8)
verde1=np.array([86,100,49], dtype=np.uint8)
verde2=np.array([100,255,255], dtype=np.uint8)
cap=cv2.VideoCapture('./videos/juzgado03.mp4')
# Roi for trafficlight juzgado03: [[(18, 220), (48, 203)]

def estadoPorFrame(imagen,vectorDoble):
	x0=vectorDoble[0][0]
	y0=vectorDoble[0][1]
	x1=vectorDoble[1][0]
	y1=vectorDoble[1][1]
	print(x0,y0,x1,y1)
	color = Find_color(imagen[y0:y1,x0:x1])
	cv2.rectangle(imagen,(x0,y0),(x1,y1),(255,255,255),1)
	return color

class Stabilizador():
	def __init__(self,initial_image, rectangulo_a_seguir):	# [[x0,y0],[x1,y1]]
		self.rectanguloX0 = rectangulo_a_seguir[0][0]
		self.rectanguloY0 = rectangulo_a_seguir[0][1]
		self.rectanguloX0 = rectangulo_a_seguir[1][0]
		self.rectanguloY1 = rectangulo_a_seguir[1][1]
		self.imagen_auxiliar_croped = initial_image[rectanguloY0:rectanguloY1,rectanguloX0:rectanguloX1]
		self.imagen_auxiliar_croped = cv2.cvtColor(np.array(self.imagen_auxiliar_croped), cv2.COLOR_BGR2GRAY)
		self.feature_parameters = dict(maxCorners = 4, qualityLevel = 0.3, minDistance = 7, blockSize= 7)
		self.lk_parameters = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
		self.colour = np.random.randint(0,255,(4,3))	# For four corners for 3 colours RGB between 0 and 255
		self.puntos_to_track = cv2.goodFeaturesToTrack(self.imagen_auxiliar_croped,mask=None,**self.feature_parameters)
		self.mask = np.zeros_like(self.imagen_auxiliar_croped)


	def obtener_vector_desplazamiento(self,nueva_Imagen):
		imagen_actual_croped = nueva_Imagen[rectanguloY0:rectanguloY1,rectanguloX0:rectanguloX1]
		imagen_actual_croped = cv2.cvtColor(np.array(imagen_actual_croped), cv2.COLOR_BGR2GRAY)
		puntos_tracked, st,err = cv2.calcOpticalFlowPyrLK(self.imagen_auxiliar_croped,imagen_actual_croped,self.puntos_to_track,**self.lk_parameters)
		good_new = puntos_tracked[st==1]
		good_old = puntos_to_track[st==1]
		vector_desplazamiento = (0,0)
		for i, (new,old) in enumerate(zip(good_new,good_old)):	# para cada punto obtienes las posiciones iniciales y finales
			a,b = new.ravel()
			c,d = old.ravel()
			vector_desplazamiento +=(c-a,d-b)
			mask = cv2.line(mask,(a,b),(c,d),self.colour[i].tolist(),2)
			frame = cv2.circle(imagen_actual_croped,(a,b),5,self.colour[i].tolist(),-1)
		vector_desplazamiento = vector_desplazamiento//4
		visualizacion = cv2.add(frame,mask)
		return visualizacion, vector_desplazamiento

	def estabilizar_imagen(self,imagen_a_estabilizar):
		filas,columnas = imagen_a_estabilizar.shape[:2]
		vector_a_desplazar = self.obtener_vector_desplazamiento(imagen_a_estabilizar)
		matriz_de_traslacion = np.float([[1,0,-vector_a_desplazar[0]],[0,1,-vector_a_desplazar[y]]])
		imagen_estabilizada = cv2.warpAffine(imagen_a_estabilizar,matriz_de_traslacion,(columnas,filas))
		return imagen_estabilizada




class CarFlowDetector():
	def __init__(self,size, vertices,angle=55):	#size (320,240)
		#Auxiliar variables
		self.auxiliar_image = np.array((0.0))
		self.w = size[0]
		self.h = size[1]
		self.size=(self.w,self.h)
		self.origenx =  size[0] // 2
		self.origeny =  size[1] // 2
		self.origen =  np.array([size[0],size[1]])
		# Class variables
		self.theta = angle
		self.optimalStep = 10

		# flow sume fx,fy va
		self.vertices = vertices

		self.total_flow_frame = np.array((0,0))

		#Auxiliares
		self.magnitudes = np.array((0.0,0.0,0.0,0.0))
		self.magnitudesFiltradas = np.array((0.0,0.0,0.0,0.0))
		self.velocidadesFiltradas = []
		self.a_coeff = np.array(( 1.,-2.37409474,1.92935567,-0.53207537))
		self.b_coeff = np.array(( 0.00289819,0.00869458,0.00869458,0.00289819))
		self.velocidades=[]

		# BackGroundSubs
		self.carSize = 800
		self.carSizeMaximum = 6000
		self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		self.back=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
		#cv2.ocl.setUseOpenCL(False)
		# Config
		self.font = cv2.FONT_HERSHEY_SIMPLEX
        
# Params for shitomasi corner dectection
#feature_params = dict( maxCorners = 10, qualityLevel = 0.3, 
#                      minDistance = 7, blockSize = 7)
	
	# Una introcucción estandar de la imagen permite cambiar al tamaño estandar de trabajo y convertirla a gray scale
	def introduce_image(self,img):
		if img.shape[:2] != self.size:
			auxiliar = cv2.resize(img,self.size)
		auxiliar = cv2.cvtColor(np.array(auxiliar), cv2.COLOR_BGR2GRAY)
		return auxiliar

	# Unicamente alteramos la imagen como auxiliar, con los estándares de normalización
	def set_previousImage(self,prev_img):
		self.auxiliar_image = self.introduce_image(prev_img)
		return self.auxiliar_image

	# Normalizamos una imagen y la adecuamos al poligono
	def enmascarar_imagen(self,image):
		image = self.introduce_image(image)	
		mask = np.zeros_like(image)
		cv2.fillPoly(mask, [self.vertices], 255)
		masked = cv2.bitwise_and(image, mask)
		return masked

	def get_visual_image(self,image):
		if image.shape[:2] != self.size:
			image = cv2.resize(image,self.size)
		return image

	def draw_vector(self,img,vector,color):
		cv2.line(img,(self.origen[0]//2,self.origen[1]//2),(self.origen[0]//2+int(vector[0]),self.origen[1]//2+int(vector[1])),color,2)
		return img

	def draw_flow(self, current_image):
		y, x = np.mgrid[self.optimalStep/2:self.h:self.optimalStep, self.optimalStep/2:self.w:self.optimalStep].reshape(2,-1)
		y = np.int32(y)
		x = np.int32(x)
		flow = cv2.calcOpticalFlowFarneback(self.auxiliar_image, current_image, None, 0.1, 3, 15, 3, 5, 1.2, 0)
		fx, fy = flow[y,x].T
		total_flow_framex = sum(fx)
		total_flow_framey = sum(fy)
		total_flow = np.array([total_flow_framex, total_flow_framey])
		module = np.sqrt(total_flow[0]**2  + total_flow[1]**2)
		lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
		lines = np.int32(lines + 0.5)
		return total_flow, module, lines
		
	def get_filtered_flow(self,vector):
		unitary_vector = np.array([np.cos(np.pi*self.theta/180),-np.sin(np.pi*self.theta/180)])
		scalar_vel = vector[0]*unitary_vector[0] + vector[1]*unitary_vector[1]
		vector_vel = scalar_vel * unitary_vector
		module_vector_vel = np.sqrt(vector_vel[0]**2 + vector_vel[1]**2)

		#Filtrando vector
		self.magnitudes[3] = self.magnitudes[2]
		self.magnitudes[2] = self.magnitudes[1]
		self.magnitudes[1] = self.magnitudes[0]
		self.magnitudes[0] = scalar_vel
		
		self.magnitudesFiltradas[3] = self.magnitudesFiltradas[2]
		self.magnitudesFiltradas[2] = self.magnitudesFiltradas[1]
		self.magnitudesFiltradas[1] = self.magnitudesFiltradas[0]
		self.magnitudesFiltradas[0] = - self.a_coeff[1]*self.magnitudesFiltradas[1]-self.a_coeff[2]*self.magnitudesFiltradas[2]-self.a_coeff[3]*self.magnitudesFiltradas[3]+self.b_coeff[0]*self.magnitudes[0]+self.b_coeff[1]*self.magnitudes[1]+self.b_coeff[2]*self.magnitudes[2]+self.b_coeff[3]*self.magnitudes[3]
		
		smooth_vector = self.magnitudesFiltradas[0]*unitary_vector

		module = np.sqrt(smooth_vector[0]**2  + smooth_vector[1]**2)
		velocidadReal = module*7.2/100 #km/h
		self.velocidades.append(self.magnitudes[0])
		self.velocidadesFiltradas.append(self.magnitudesFiltradas[0])

		return smooth_vector, velocidadReal

	def getCarCenters(self,current_image):
		vis = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
		fram = cv2.medianBlur(vis,7) # 7 default
		fgmask = self.back.apply(fram)
		blur = cv2.GaussianBlur(fgmask,(7,7),1)
		fgmask = cv2.morphologyEx(blur, cv2.MORPH_OPEN, self.kernel)
		contor = fgmask.copy()
		im, contours,hierarchy =cv2.findContours(contor,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		centers = []
		rectangles = []
		for cantObjetos in contours:
			if (cv2.contourArea(cantObjetos) >= self.carSize) & (cv2.contourArea(cantObjetos) <= self.carSizeMaximum):
			#if cv2.contourArea(cantObjetos) >=self.carSize:
				x,y,ancho,alto=cv2.boundingRect(cantObjetos)
				#cv2.rectangle(vis,(x,y),(x+ancho,y+alto),(0,0,255),2)
				a=int(x+ancho/2)
				b=int(y+alto/2)
				#cv2.imshow('imagen',vis) 
				centers.append([a,b])
				rectangles.append([x,y,ancho,alto])
		return np.array(centers), np.array(rectangles)

def __main_function__(fn):
	#Fur 240x360
	#original_image.shape[1]//2-10,0],[50,original_image.shape[0]],[original_image.shape[1]//2-5,original_image.shape[0]],[original_image.shape[1]//2+10,0]])
	##size (320,240)
        # Loading variables and setting
	variables = np.load('datos.npy')
	semaforo = variables[0][0]
	vertices = variables[0][1]
	angulo = variables[0][2]
	#is_red = _traffic_light_is_red()
	is_red = True

	logImagenes = []
	contadorParaVideos = 0

	distanciaRecorridaAuto = 0
	determinacion = ' no cruzo'
	cruzo = False
	guardado = False

	# Create objects
	myFlowMahine = CarFlowDetector((320,240), np.array([vertices]) , angulo)

	# Configure camera
	#cam = cv2.VideoCapture(0)
	#ret, picture = cam.read()
	vs = WebcamVideosStream(scr=0).start()
	fps = FPS().start()
	picture = vs.read()
	picture = imutils.resize(picture)

	# Config
	myFlowMahine.set_previousImage(picture)
	fps.stop()

	while True:
		#ret, img = cam.read()	# Leo imagen
		img = vs.read()
		img = imutils.resize(img)
		fps = FPS().start()
		contadorParaVideos+=1
		if contadorParaVideos%10==1:
			logImagenes.append(img) # Guardo en RAM
			visualizacion = myFlowMahine.get_visual_image(img)	# Visualizacion
			imagenRecortada = myFlowMahine.enmascarar_imagen(img) # Acondiciono
			is_red = estadoPorFrame(visualizacion,semaforo)
			if is_red == 0:
				colorLiteral = 'verde'
			if is_red == 1:
				colorLiteral = 'rojo'
			if is_red == 2:
				colorLiteral = 'amarillo'
			if is_red == 3:
				colorLiteral = 'No Semaforo'
			name = str(colorLiteral)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(visualizacion,name,(270,10), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

			if True:#is_red ==1:

				total_flow, module, lines = myFlowMahine.draw_flow(imagenRecortada) # Obtengo vector y parametros de flujo
				vectorSuave, velocidad = myFlowMahine.get_filtered_flow(total_flow)	# Proyecto y filtroFiltro 
				centros, rectangulos = myFlowMahine.getCarCenters(imagenRecortada)	#Obtengo centros y sus parámetros
				#vis = visualizacion#cv2.cvtColor(imagenRecortada, cv2.COLOR_GRAY2BGR)
				cv2.polylines(visualizacion, lines, 0, (255, 55, 0))
				distanciaRecorridaAuto +=velocidad/10
				if velocidad<0.7:
					distanciaRecorridaAuto = 0
				if distanciaRecorridaAuto >= 5:
					determinacion = ' cruzo en rojo'
					cruzo = True
				else:
					determinacion = ' no cruzo'
					if cruzo == True:
						guardado = True
					cruzo = False

				print(velocidad)
				#print(centros)
				for rectangulo in rectangulos:
					cv2.rectangle(visualizacion,(rectangulo[0],rectangulo[1]),(rectangulo[0]+rectangulo[2],rectangulo[1]+rectangulo[3]),(127,127,0),3)
				vel_label = str(int(velocidad))+' km/h'+str(int(distanciaRecorridaAuto))+'m'+determinacion
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(visualizacion,vel_label,(0,myFlowMahine.h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
				visualizacion = myFlowMahine.draw_vector(visualizacion,vectorSuave,(255,255,255))
			myFlowMahine.set_previousImage(cv2.cvtColor(imagenRecortada, cv2.COLOR_GRAY2BGR))
			#myFlowMahine.auxiliar_image = imagenRecortada
			
			cv2.imshow('flow', visualizacion )
			fps.update()
			#cv2.imshow('vista', visualizacion )
			#p0 = flow.reshape(-1,1,2)
			#p0 = (cv2.goodFeaturesToTrack(prevgray, mask = None, **feature_params)).reshape(-1,1,2)

		else:
			if False: #guardado
				print("Enviando Infraccion")
				for index in range(len(logImagenes)):
					cv2.imwrite('imagen_{}.jpg'.format(str(index)),logImagenes[index])
				logImagenes = []
		ch = 0xFF & cv2.waitKey(5)
		#is_red = findColor()
		if ch == 27:
			break
		fps.stop()
		picture,release()
		cv2.destroyAllWindows()
	__main_function__(fn)

if __name__ == '__main__':
    
    import sys
    print (help_message)
    try: fn = sys.argv[1]

    except: fn = 0
    __main_function__(fn)
