#!/usr/bin/env python
#
# The is a port of SmoothLife to python. 
#
import numpy as np
from scipy import weave
from time import sleep
from random import random
# Display modules
import matplotlib.pyplot as plt


def main():
	NX = 400
	NY = 400
	ra = 10
	ri = ra/3
	b = 1
	b1 = 0.278
	b2 = 0.365
	d1 = 0.267
	d2 = 0.445
	alphan = 0.028
	alpham = 0.147

	kd = np.zeros((NY,NX))
	kr = np.zeros((NY,NX))
	#aa = np.zeros(NY,NX)
	for iy in range(1,NY):
		for ix in range(1,NX):
			x = ix-1-NX/2
			y = iy-1-NY/2
			r = np.sqrt(x*x+y*y)
			kd[iy,ix] = 1- func_linear(r, ri, b)
			kr[iy,ix] = func_linear(r, ri, b)*(1-func_linear(r, ra, b))
			#aa[iy,ix] = snm(ix/NX, iy/NY, alphan, alpham, b1, b2, d1, d2)
	
	
	kflr = sum(sum(kr))
	kfld = sum(sum(kd))
	krf = np.fft.fft2(np.fft.fftshift(kr))
	kdf = np.fft.fft2(np.fft.fftshift(kd))

	# Generate Initial Data
	#aa = init_circle(NY,NX,ra)
	aa = initaa(NY,NX,ra)

	# Initialize figure
	plt.ion()
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111)
	fig.show()
	plt.axis('off')

	
	while True:
		aaf = np.fft.fft2(aa)
		nf = aaf*krf
		mf = aaf*kdf

		n = np.real(np.fft.ifft2(nf))/kflr
		m = np.real(np.fft.ifft2(mf))/kfld
		aa = snm(n, m, alphan, alpham, b1, b2, d1, d2)
		
		ax.cla()
		#ax.imshow(aa, cmap=plt.cm.gray)
		ax.imshow(aa)
		plt.axis('off')
		fig.canvas.draw()
		#sleep(0.05)

		#break if everything is dead
		#if round(sum(sum(aa))) == 0.0: break

	raw_input("All is dead!")



def func_linear(x, a, b):
	if x < a-b/2:
		return 0
	elif x > a+b/2:
		return 1
	else:
		return (x-a+b/2)/b

def func_smooth(x, a, b):
	return 1./(1+np.exp(-(x-a)*4/b))

def sigmoid_a(x, a, ea):
	return func_smooth(x, a, ea)

def sigmoid_b(x, b, eb):
	return 1 - sigmoid_a(x, b, eb)

def sigmoid_ab(x, a, b, ea, eb):
	return sigmoid_a(x, a, ea)*sigmoid_b(x, b, eb)

def sigmoid_mix(x, y, m, em):
	return x * (1- func_smooth(m, 0.5, em)) + (y * func_smooth (m, 0.5, em))

def snm(n, m, en, em, b1, b2, d1, d2):
	return sigmoid_ab(n, sigmoid_mix(b1, d1, m, em), sigmoid_mix(b2, d2, m, em), en, en)

def splat(aa, ny, nx, ra):
	x = np.floor(random()*nx)+1
	y = np.floor(random()*ny)+1
	if random()>0.5:
		c = 1
	else:
		c = 0

	for dx in range(-ra,ra-1):
		for dy in range(-ra,ra-1):
			ix = x+dx
			iy = y+dy
			if ix>=0 and ix<=nx-1 and iy>=0 and iy<=ny-1:
				aa[iy,ix] = c

	return aa

def initaa(ny, nx, ra):
	aa = np.zeros((ny,nx))
	for t in range(0, ((nx/ra)*(ny/ra))):
		aa = splat(aa, ny, nx, ra)
	
	return aa

def init_circle(ny,nx,ra):
	aa = np.zeros((ny,nx))
	cent_y = ny/2
	cent_x = nx/2
	radius = min([cent_y,cent_x])/2

	for t in np.arange(0,2*np.pi,0.01):
		ix = np.floor(cent_y + radius*np.cos(t))
		iy = np.floor(cent_x + radius*np.sin(t))
		aa[iy,ix] = 1

	return aa

def init_diag(ny,nx,ra):
	aa = np.zeros((ny,nx))
	return aa

if __name__ == '__main__':
	main()