# Part 3: Group assignments
In a hypothetical mid-western university, the staff of a CS course randomly assigns students to project
teams, but does so based on some input from the students. In particular, each student answers the following
questions:
1. What is your user ID?
2. Would you prefer to work alone, in a team of two, in a team of three, or do you not have a preference? Please enter 1, 2, 3, or 0, respectively.
3. Which student(s) would you prefer to work with? Please list their user IDs separated by commas, or leave this box empty if you have no preference.
4. Which students would you prefer not to work with? Please list their IDs, separated by commas.

Unfortunately, student preferences often conflict with one another, so that it is not possible to find an
assignment that makes everyone happy. Instead, the course staff tries to find an assignment that will
minimize the amount of work they have to do after assigning the groups. They estimate that:

1. They need k minutes to grade each assignment, so total grading time is k times number of teams.
2. Each student who requested a specific group size and was assigned to a different group size will complain to the instructor after class, taking 1 minute of the instructor's time.
3. Each student who is not assigned to someone they requested will send a complaint email, which will take n minutes for the instructor to read and respond. If a student requested to work with multiple people, then they will send a separate email for each person they were not assigned to. 
4. Each student who is assigned to someone they requested not to work with (in question 4 above) will request a meeting with the instructor to complain, and each meeting will last m minutes. If a student
requested not to work with two specific students and is assigned to a group with both of them, then
they will request 2 meetings.

The total time spent by the course staff is equal to the sum of these components. Your goal is to write a
program to find an assignment of students to teams that minimizes the total amount of work the course staff
needs to do, subject to the constraint that no team may have more than 3 students. Your program should
take as input a text file that contains each student's response to these questions on a single line, separated
by spaces. For example, a sample file might look like:

djcran 3 zehzhang,chen464 kapadia

chen464 1 _ _

fan6 0 chen464 djcran

zehzhang 1 _ kapadia

kapadia 3 zehzhang,fan6 djcran

steflee 0 _ _

where the underscore character ( ) indicates an empty value. Your program should be run like this:
> ./assign.py [input-file] [k] [m] [n]

where k, m, and n are values for the parameters mentioned above, and the output of your program should
be a list of group assignments, one group per line, with user names separated by spaces followed by the total
time requirement for the instructors, e.g.:

djcran chen464

kapadia zehzhang fan6

steflee

534

which proposes three groups, with two people, three people, and one person, respectively.
