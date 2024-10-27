a = int(input('Gimme a positive integer '))

nums = []
for i in range(a):
	b = input('Gimme a number ')
	nums.append(int(b))

c = int(input('What number are you looking for? '))

try:
	print(nums.index(c) + 1)
except: 
	print('-1')



