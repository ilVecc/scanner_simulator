#
# easy way to get everything started properly :)
#
__import__("ensurepip").bootstrap(upgrade=True)
__import__("pip").main("install -r requirements.txt".split())
