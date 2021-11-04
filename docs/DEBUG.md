# Debugging Checklist

Following failures, please use the below checklist to create a github issue:

Date / time / task / job board as title of the issue

Summary of what you think the error means

Paste the entire stack trace as a comment in the issue

Debug step a)

- [ ] Try rerunning in test mode - does it work?

- [ ] If yes, try rerunning in production mode - does it work?

Debug step b)

- [ ] If not, then isolate the problem and raise another issue, linking to here. Do not close this issue. Consider removing the task from cron.

- [ ] Once the problem is isolated, repeat a) and b) until resolved.

Debug step c)

- [ ] See if the next cron runs fine - does it work?

- [ ] If yes, conclude with remedial steps

- [ ] If not, start over
