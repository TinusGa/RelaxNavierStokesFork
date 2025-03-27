
Ownership_ranges =  [   0,  440 , 968 ,1408, 1936 ,2376 ,2904 ,3344 ,3872]

for i in range(1,len(Ownership_ranges)):
    last_el = Ownership_ranges[-i]
    next_el = Ownership_ranges[-(i+1)]
    diff = last_el - next_el
    print(f"diff: {diff}")


custom_tuple = ((1,2),(3,4))

print(custom_tuple[1][0])
