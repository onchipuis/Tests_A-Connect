import os
import sys
folder = os.getcwd()

sys.path.append(folder)

for i in range(5):
	if i == 0:
		config = open('config.txt','w')
		config.write(folder)
		config.close()
	elif i == 1:
		config = open('./Layers/config.txt','w')
		config.write(folder)
		config.close()
	elif i == 2:
		config = open('./Networks/config.txt','w')
		config.write(folder)
		config.close()		
	elif i == 3:
		config = open('./Scripts/config.txt','w')
		config.write(folder)
		config.close()
	elif i == 4:
		config = open('./Images/config.txt','w')
		config.write(folder)
		config.close()		

