import data_class
from datetime import date


g = data_class.DataField()
f = data_class.DataField()

g.load_station_data('Klemday07.raw')
f.load_station_data('TG_STAID000027.txt', dataset = 'ECA-station')

print g.location, g.data.shape
print f.location, f.data.shape

# compare
d1, m1, y1 = g.extract_day_month_year()
wrong = []
cnt = 0
for i in range(g.data.shape[0]):
    if g.time[i] == f.time[i]:
        if g.data[i] != f.data[i]:
            cnt += 1
            wrong.append(date(y1[i], m1[i], d1[i]))
    else:
        raise 'Something went terribly wrong'