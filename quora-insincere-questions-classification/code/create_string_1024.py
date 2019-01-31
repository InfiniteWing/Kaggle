def get_encode_str():
    MAX_LEN = 1024
    UNIQS = []
    i = 20000
    while len(UNIQS)< MAX_LEN:
        try:
            str(chr(i))
            UNIQS.append(chr(i))
        except UnicodeEncodeError:
            pass
        i += 1
    print(''.join(UNIQS))
get_encode_str()