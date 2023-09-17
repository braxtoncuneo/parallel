import os
import re

cwd = os.getcwd()

src_path = os.path.join(cwd,"src")

modes = [ "none", "web", "slide", "both" ]


path_list = {}


slide_style  = "<style>\n"
slide_style += ":root { --content-max-width: 90%; }\n"
slide_style += "body { font-size: 4rem; }\n"
slide_style += ".segment_container { display: flex; flex-direction: row; }\n"
slide_style += ".segment { margin-left: 2%; margin-right: 2%; flex: 1; }\n"
slide_style += "</style>\n"


cell_start = "<div class=\"segment\">\n\n"
cell_end   = "</div>\n"

row_start = "<div class=\"segment_container\">\n"
row_start += cell_start

row_end = "</div>\n"
row_end = cell_end + row_end

slide_start = slide_style + row_start
slide_end   = row_end

def segment_path(path):
    seq = []
    segment = path
    while True:
        head, tail = os.path.split(segment)
        segment = head

        if str(tail) != "":
            seq.append(tail)
        elif str(head) != "":
            seq.append(head)
        else:
            break
    seq.reverse()
    return seq



def generate_slide(slide_content,slide_number,path,file):
    slide_name = f"{slide_number}"
    file_relpath = os.path.relpath(path,start=src_path)
    file_relpath, _ = os.path.splitext(file_relpath)

    seq = segment_path(file_relpath)

    base = path_list
    for p in seq:
        if p not in base:
            base[p] = {}
        base = base[p]
    base[slide_name] = {}

    base_path    = os.path.join('src','slide')
    file_relpath = os.path.join(base_path,file_relpath)
    os.makedirs(file_relpath, exist_ok=True)

    slide_path = os.path.join(file_relpath,slide_name)
    slide_file = open(slide_path,mode='w')
    slide_file.write(slide_content)
    slide_file.close()

def process_file(subdir,file):
    path = os.path.join(subdir,file)

    slide_number = 0

    current_mode = 3

    file_in = open(path)
    web_content   = ""
    slide_content = slide_start
    for line in file_in:
        directive = r"^[ \t]*<!--[ \t]*slider[ \t]*(?P<dir>[\w-]+)[ \t]*-->[ \t]*$"
        match = re.match(directive,line)
        if match != None:
            d = match.group('dir')
            if d in modes:
                index = modes.index(d)
                current_mode = index
                mode_name = modes[current_mode]
                line = ""
            elif d == "row-split" :
                slide_content += row_end
                slide_content += row_start
            elif d == "cell-split" :
                slide_content += cell_end
                slide_content += cell_start
            elif d == "split" :
                slide_content += slide_end
                generate_slide(slide_content,slide_number,path,file)
                slide_content = slide_start
                slide_number  += 1
        
        web_relpath = os.path.relpath(subdir,start=src_path)
        depth = len(segment_path(web_relpath))
        relpath = r"{{[ \t]*#relpath[ \t]*}}"
        slide_relpath = "../.."
        for _ in range(depth):
            slide_relpath = os.path.join(slide_relpath,"..")
        slide_relpath = os.path.join(slide_relpath,web_relpath)

        line = re.sub(relpath,str(slide_relpath),line)
        if current_mode // 2 == 1:
            slide_content += line
    
    slide_content += slide_end
    generate_slide(slide_content,slide_number,path,file)
    slide_content  = ""
    slide_number  += 1

    file_in.close()


for subdir, dirs, files in os.walk(src_path):
    for file in files:
        if not file.endswith(".md"):
            continue

        process_file(subdir,file)




def slide_TOC(base,prefix='./slide',depth=1):
    result = ""
    for parent, children in base.items():
        tabs = "\t"*depth
        if children :
            result += f"{tabs}- [{parent}]()\n"
            result += slide_TOC(children,f"{prefix}/{parent}",depth+1)
        else:
            result +=f"{tabs}- [{parent}]({prefix}/{parent})\n"
    return result




TOC = open("./src/TOC.md")
TOC_text = TOC.read()
TOC.close()

print(path_list)

summary = open("./src/SUMMARY.md",mode="w")
summary.write(TOC_text)
summary.write("- [Slides]()\n")
summary.write(slide_TOC(path_list))
summary.close()



