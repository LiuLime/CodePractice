from datetime import date, timedelta

start_date = date(2010, 1, 1)
day_of_week_start = 3
friday_13th_count = 0

current_date = start_date
# print(current_date, "星期", day_of_week_start + 1)
while friday_13th_count < 667:
    if current_date.day == 13 and day_of_week_start == 4:
        print(current_date)
        friday_13th_count += 1
    if friday_13th_count == 666:
        # print(current_date)
        result_date = current_date.strftime('%Y%m%d')
        # print(result_date)
        break

    day_of_week_start = (day_of_week_start + 1) % 7
    # Move to the next day
    current_date += timedelta(days=1)
    # print(current_date, "星期", day_of_week_start + 1)
    # print(current_date)
