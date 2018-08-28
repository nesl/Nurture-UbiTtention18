import numpy as np
import itertools

# csv file format:
# <question_statement>, <hour>, <minute>, <day>, <motion_activity>, <location>, <last_notification_time>


# Script paramteres
OUT_FILENAME = 'data/02_mar11_p1_1000.csv'
NUM_QUESTIONS = 1000


def get_time():
    # 8am-10pm with an interval of 15 minutes
    hour = np.random.randint(14) + 8
    minute = np.random.choice([0, 15, 30, 45])
    am_pm = 'AM' if hour < 12 else 'PM'
    hour_12 = hour if hour <= 12 else hour - 12
    minute_text = "" if minute == 0 else ":%d" % minute
    text = "%d%s %s" % (hour_12, minute_text, am_pm)
    return (text, hour, minute)

def get_day():
    day = np.random.randint(7)
    day_words = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return (day_words[day], day)

def choose_tuple(data):
    idx = np.random.randint(len(data))
    return data[idx]

def template_stationary():
    location_label_text = [
        ('home', 'home'),
        ('work', 'work'),
        ('market', 'market'),
        ('mall', 'a shopping mall'),
        ('friend-house', 'your friend\'s place'),
    ]

    location_label, location_text = choose_tuple(location_label_text)
    full_sentence = "You're sitting (or standing) at %s." % location_text
    return (full_sentence, 'stationary', location_label)

def template_moving_large_distance():
    motion_label_text = [
        ('walking', 'walking'),
        ('driving', 'driving'),
        ('bus', 'taking a bus'),
        ('train', 'taking a train'),
    ]

    motion_label, motion_text = choose_tuple(motion_label_text)
    full_sentence = "You are %s to some place." % motion_text
    return (full_sentence, motion_label, 'others')

def template_moving_small_distnce():
    location_label_text = [
        ('home', 'your home'),
        ('work', 'your working place'),
        ('market', 'a market'),
        ('mall', 'a shopping mall'),
        ('friend-house', 'your friend\'s place'),
    ]
    
    location_label, location_text = choose_tuple(location_label_text)
    full_sentence = "You are walking around %s." % location_text
    return (full_sentence, 'walking', location_label)

def template_stationary_question():
    location_label_text = [
        ('home', 'home'),
        ('work', 'your working place'),
        ('restaurant', 'a restaurant'),
        ('market', 'a market'),
        ('movie-theater', 'a movie theater'),
        ('friend-house', 'your friend\'s place'),
    ]

    location_label, location_text = choose_tuple(location_label_text)
    full_sentence = "You are not moving (you could be sleeping or sitting some place). You are at %s." % location_text
    return (full_sentence, 'stationary', location_label)

def template_physical_question():
    motion_label_text = [
        ('walking', 'walking'),
        ('running', 'running'),
        ('biking', 'biking'),
    ]
    location_label_text = [
        ('beach', 'the beach'),
        ('gym', 'a gym'),
        ('park', 'a park'),
    ]

    motion_label, motion_text = choose_tuple(motion_label_text)
    location_label, location_text = choose_tuple(location_label_text)
    full_sentence = "You are %s at %s." % (motion_text, location_text)
    return (full_sentence, motion_label, location_label)

def get_motion_location():
    template = np.random.choice(
            [template_stationary, template_moving_large_distance, template_moving_small_distnce, template_stationary_question, template_physical_question],
            p=[0.18,              0.13,                           0.23,                          0.23,                         0.23],
    )
    return template()

def pluralize(value, unit):
    if value == 0:
        return ""
    elif value == 1:
        return "1 %s" % unit
    else:
        return "%d %ss" % (value, unit)

def get_last_notification_time():
    ranges = [range(1, 5), range(5, 60, 5), range(60, 120, 15), range(120, 240, 30), range(240, 300, 60), range(9999, 10000)]
    time_groups = [list(param) for param in ranges]
    times = list(itertools.chain.from_iterable(time_groups))

    time_elasped = np.random.choice(times)
    if time_elasped >= 9999:
        return ("more than 5 hours", 9999)
    else:
        hour, minute = divmod(time_elasped, 60)
        hour_text = pluralize(hour, 'hour')
        minute_text = pluralize(minute, 'minute')
        if hour_text != "" and minute_text != "":
            full_description = "%s and %s" % (hour_text, minute_text)
        else:
            full_description = "%s%s" % (hour_text, minute_text)
        return (full_description, time_elasped)

def generate_csv_row():
    time_text, hour_label, minute_label = get_time()
    day_text, day_label = get_day()
    motion_location_text, motion_label, location_label = get_motion_location()
    last_notification_time_text, last_notification_time_label = get_last_notification_time()

    question_statement = " ".join([
            "It is %s on %s." % (time_text, day_text),
            "%s" % motion_location_text,
            "You responded to (or clicked on) a notification %s ago." % last_notification_time_text,
            "Now you receive a notification from our app.",
            "The notification says \"\"Can you take 10 seconds to complete this questionnaire?\"\"",
            "What will you do?", 
    ])
    line = ",".join(list(map(str, [
            "\"%s\"" % question_statement,
            hour_label,
            minute_label,
            day_label,
            motion_label,
            location_label,
            last_notification_time_label,
    ])))
    return line + "\n"


def main():
    out_filename = OUT_FILENAME
    with open(out_filename, "w") as fo:
        fo.write("content,hour,minute,day,motion,location,last_notification_time\n")
        for i in range(NUM_QUESTIONS):
            fo.write(generate_csv_row())


if __name__ == "__main__":
    main()
