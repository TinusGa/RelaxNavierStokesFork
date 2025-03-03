filename = "prints.txt"
with open(filename, "r") as file:
    content = file.read()
lines = content.split("\n")
line_count = len(lines)
print("Number of lines:", line_count)
