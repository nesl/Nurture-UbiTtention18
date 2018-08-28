"""
The csv file format is the similar to the 2nd version of the survey `ver2_mturk`, except that we
add the last column to indicate number of days since the simulation starts. This new column will
not be part of the the question statement generation process, however, it helps restoring the
survey question generation process.

csv file format:
<question_statement>, <hour>, <minute>, <day>, <motion_activity>, <location>, <last_notification_time>, <num_days_passed>
"""

import csv
import numpy as np


class MTurkSurveyGenerator:

    LABEL_NOTIF_TIME_MORE_THAN_5_HOURS = 9999

    def __init__(self, out_file_name):
        self.out_file_name = out_file_name
        self.motion_location_dict = self._create_motion_location_dictionary()

    def __enter__(self):
        self.fo = open(self.out_file_name, 'w', newline='')
        fieldnames = ['content', 'hour', 'minute', 'day', 'motion', 'location', 'last_notification_time', 'num_days_passed']
        self.writer = csv.DictWriter(self.fo, fieldnames=fieldnames)
        self.writer.writeheader()
        return self

    def __exit__(self, excType, excValue, excTraceback):
        if excType is not None:
            raise Exception("An error occur in MTurkSurveyGenerator")
        
        self.fo.close()

    def add_row(self, hour_label, minute_label, day_label, motion_label, location_label,
            last_notification_time_label, num_days_passed):
        time_text = self._get_time(hour_label, minute_label)
        day_text = self._get_day(day_label)
        motion_location_text = self._get_motion_location(motion_label, location_label)
        last_notification_time_text = self._get_last_notification_time(last_notification_time_label)

        question_statement = " ".join([
                "It is %s on %s." % (time_text, day_text),
                "%s" % motion_location_text,
                "You responded to (or clicked on) a notification %s ago." % last_notification_time_text,
                "Now you receive a notification from our app.",
                "The notification says \"Can you take 10 seconds to complete this questionnaire?\"",
                "What will you do?", 
        ])

        self.writer.writerow({
                'content': question_statement,
                'hour': hour_label,
                'minute': minute_label,
                'day': day_label,
                'motion': motion_label,
                'location': location_label,
                'last_notification_time': last_notification_time_label,
                'num_days_passed': num_days_passed,
        })

    def _get_time(self, hour, minute):
        am_pm = 'AM' if hour < 12 else 'PM'
        hour_12 = hour if hour <= 12 else hour - 12
        minute_text = "" if minute == 0 else ":%d" % minute
        text = "%d%s %s" % (hour_12, minute_text, am_pm)
        return text

    def _get_day(self, day):
        return ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][day]

    def _get_motion_location(self, motion, location):
        ml = (motion, location)
        if ml not in self.motion_location_dict:
            raise Exception("(motion=%s, location=%s) does not have the corresponding statement."
                    % (motion, location))
        return self.motion_location_dict[ml]()
    
    def _create_motion_location_dictionary(self):
        # create a dictionary of (motion, location) => statement
        return {
                ('stationary', 'home'): lambda: np.random.choice([
                        "You're sitting (or standing) at home.",
                        "You are not moving (you could be sleeping or sitting some place). You are at home.",
                ]),
                ('stationary', 'work'): lambda: np.random.choice([
                        "You're sitting (or standing) at work.",
                        "You are not moving (you could be sleeping or sitting some place). You are at your working place.",
                ]),
                ('stationary', 'others'): lambda: np.random.choice([
                        "You're sitting (or standing) at a market.",
                        "You're sitting (or standing) at a shopping mall.",
                        "You're sitting (or standing) at your friend's place.",
                        "You are not moving (you could be sleeping or sitting some place). You are at a restaurant.",
                        "You are not moving (you could be sleeping or sitting some place). You are in a market.",
                        "You are not moving (you could be sleeping or sitting some place). You are at a movie theater.",
                        "You are not moving (you could be sleeping or sitting some place). You are at your friend's place.",
                ]),
                ('walking', 'home'): lambda: np.random.choice([
                        "You are walking around your home.",
                ]),
                ('walking', 'work'): lambda: np.random.choice([
                        "You are walking around your working place.",
                ]),
                ('walking', 'others'): lambda: np.random.choice([
                        "You are walking to some place.",
                        "You are walking around a shopping mall.",
                        "You are walking around your friend's place.",
                        "You are walking at the beach.",
                        "You are walking at a gym.",
                        "You are walking in a park.",
                    ], p=[
                        0.33,
                        0.17,
                        0.17,
                        0.11,
                        0.11,
                        0.11,
                    ],
                ),
                ('running', 'others'): lambda: np.random.choice([
                        "You are running at the beach.",
                        "You are running at a gym.",
                        "You are running in a park.",
                ]),
                ('driving', 'others'): lambda: np.random.choice([
                        "You are driving to some place.",
                        "You are biking at the beach.",
                        "You are biking at a gym.",
                        "You are biking at a park.",
                    ], p=[
                        0.7,
                        0.1,
                        0.1,
                        0.1,
                    ],
                ),
                ('commuting', 'others'): lambda: np.random.choice([
                        "You are taking a bus to some place.",
                        "You are taking a train to some place.",
                ]),
        }

    def _pluralize(self, value, unit):
        if value == 0:
            return ""
        elif value == 1:
            return "1 %s" % unit
        else:
            return "%d %ss" % (value, unit)

    def _get_last_notification_time(self, time_elapsed):
        if time_elapsed == MTurkSurveyGenerator.LABEL_NOTIF_TIME_MORE_THAN_5_HOURS:
            return "more than 5 hours"
        else:
            hour, minute = divmod(time_elapsed, 60)
            hour_text = self._pluralize(hour, 'hour')
            minute_text = self._pluralize(minute, 'minute')
            if hour_text != "" and minute_text != "":
                return "%s and %s" % (hour_text, minute_text)
            else:
                return "%s%s" % (hour_text, minute_text)
