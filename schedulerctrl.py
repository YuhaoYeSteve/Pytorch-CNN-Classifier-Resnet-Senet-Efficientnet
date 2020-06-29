

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler

jobs = []
def shutdown():
	for job in jobs:
		job.shutdown()

def start_scheduler(**kwargs):
	"""
	启动一个定时任务，定时执行某些任务
	:param kwargs: 
	add_job(func, trigger=None, args=None, kwargs=None, id=None, \
		name=None, misfire_grace_time=undefined, coalesce=undefined, \
		max_instances=undefined, next_run_time=undefined, \
		jobstore='default', executor='default', \
		replace_existing=False, **trigger_args)

	Adds the given job to the job list and wakes up the scheduler if it's already running.

	Any option that defaults to ``undefined`` will be replaced with the corresponding default
	value when the job is scheduled (which happens when the scheduler is started, or
	immediately if the scheduler is already running).

	The ``func`` argument can be given either as a callable object or a textual reference in
	the ``package.module:some.object`` format, where the first half (separated by ``:``) is an
	importable module and the second half is a reference to the callable object, relative to
	the module.

	The ``trigger`` argument can either be:
	  #. the alias name of the trigger (e.g. ``date``, ``interval`` or ``cron``), in which case
		any extra keyword arguments to this method are passed on to the trigger's constructor
	  #. an instance of a trigger class

	:param func: callable (or a textual reference to one) to run at the given time
	:param str|apscheduler.triggers.base.BaseTrigger trigger: trigger that determines when
		``func`` is called
	:param list|tuple args: list of positional arguments to call func with
	:param dict kwargs: dict of keyword arguments to call func with
	:param str|unicode id: explicit identifier for the job (for modifying it later)
	:param str|unicode name: textual description of the job
	:param int misfire_grace_time: seconds after the designated runtime that the job is still
		allowed to be run
	:param bool coalesce: run once instead of many times if the scheduler determines that the
		job should be run more than once in succession
	:param int max_instances: maximum number of concurrently running instances allowed for this
		job
	:param datetime next_run_time: when to first run the job, regardless of the trigger (pass
		``None`` to add the job as paused)
	:param str|unicode jobstore: alias of the job store to store the job in
	:param str|unicode executor: alias of the executor to run the job with
	:param bool replace_existing: ``True`` to replace an existing job with the same ``id``
		(but retain the number of runs from the existing one)
	:rtype: Job
	:return: 
	"""
	scheduler = BlockingScheduler(timezone="Asia/Shanghai")
	scheduler.add_job(**kwargs)

	jobs.append(scheduler)
	scheduler.start()


def start_background_scheduler(**kwargs):
	"""
	启动一个定时任务，定时执行某些任务
	:param kwargs: 
	add_job(func, trigger=None, args=None, kwargs=None, id=None, \
		name=None, misfire_grace_time=undefined, coalesce=undefined, \
		max_instances=undefined, next_run_time=undefined, \
		jobstore='default', executor='default', \
		replace_existing=False, **trigger_args)

	Adds the given job to the job list and wakes up the scheduler if it's already running.

	Any option that defaults to ``undefined`` will be replaced with the corresponding default
	value when the job is scheduled (which happens when the scheduler is started, or
	immediately if the scheduler is already running).

	The ``func`` argument can be given either as a callable object or a textual reference in
	the ``package.module:some.object`` format, where the first half (separated by ``:``) is an
	importable module and the second half is a reference to the callable object, relative to
	the module.

	The ``trigger`` argument can either be:
	  #. the alias name of the trigger (e.g. ``date``, ``interval`` or ``cron``), in which case
		any extra keyword arguments to this method are passed on to the trigger's constructor
	  #. an instance of a trigger class

	:param func: callable (or a textual reference to one) to run at the given time
	:param str|apscheduler.triggers.base.BaseTrigger trigger: trigger that determines when
		``func`` is called
	:param list|tuple args: list of positional arguments to call func with
	:param dict kwargs: dict of keyword arguments to call func with
	:param str|unicode id: explicit identifier for the job (for modifying it later)
	:param str|unicode name: textual description of the job
	:param int misfire_grace_time: seconds after the designated runtime that the job is still
		allowed to be run
	:param bool coalesce: run once instead of many times if the scheduler determines that the
		job should be run more than once in succession
	:param int max_instances: maximum number of concurrently running instances allowed for this
		job
	:param datetime next_run_time: when to first run the job, regardless of the trigger (pass
		``None`` to add the job as paused)
	:param str|unicode jobstore: alias of the job store to store the job in
	:param str|unicode executor: alias of the executor to run the job with
	:param bool replace_existing: ``True`` to replace an existing job with the same ``id``
		(but retain the number of runs from the existing one)
	:rtype: Job
	:return: 
	"""
	scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
	scheduler.add_job(**kwargs)
	jobs.append(scheduler)
	scheduler.start()