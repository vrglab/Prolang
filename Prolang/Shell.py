import LangMain

while True:
    text = input('>>> ')
    result,error = LangMain.Run('<stdin>',text)

    if error: print(error.as_string())
    else: print(result)
	
