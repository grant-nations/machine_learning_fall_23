def pretty_print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key) + ':')
            pretty_print_dict(value, indent + 1)
        elif isinstance(value, list):
            print('  ' * indent + str(key) + ':')
            for item in value:
                if isinstance(item, dict):
                    pretty_print_dict(item, indent + 2)
                else:
                    print('  ' * (indent + 1) + str(item))
        else:
            print('  ' * indent + str(key) + ': ' + str(value))