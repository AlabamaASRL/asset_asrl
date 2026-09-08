# -*- coding: utf-8 -*-

"""
Astronomical date and Julian Date conversion utilities.

This module provides conversions between calendar dates, Julian Dates (JD),
Modified Julian Dates (MJD), fractional days, and datetime objects.

Python's datetime module assumes a proleptic Gregorian calendar. The Julian
and Gregorian calendars are treated separately by the Julian Date conversion
functions in this module.

The Gregorian calendar took effect on October 15, 1582. The dates October 5
through October 14, 1582 did not occur in regions that adopted the original
Gregorian transition.
"""

import datetime as dt
import math


SECONDS_PER_DAY = 86400
MICROSECONDS_PER_SECOND = 1_000_000
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * MICROSECONDS_PER_SECOND
MICROSECONDS_PER_MINUTE = 60 * MICROSECONDS_PER_SECOND
MICROSECONDS_PER_HOUR = 3600 * MICROSECONDS_PER_SECOND

JD_MJD_OFFSET = 2400000.5
JD_J2000 = 2451545.0


def mjd_to_jd(mjd):
    """
    Convert Modified Julian Day to Julian Day.

    Parameters
    ----------
    mjd : float
        Modified Julian Day.

    Returns
    -------
    float
        Julian Day.
    """
    return mjd + JD_MJD_OFFSET


def jd_to_mjd(jd):
    """
    Convert Julian Day to Modified Julian Day.

    Parameters
    ----------
    jd : float
        Julian Day.

    Returns
    -------
    float
        Modified Julian Day.
    """
    return jd - JD_MJD_OFFSET


def date_to_jd(year, month, day):
    """
    Convert a calendar date to Julian Day.

    Algorithm from:

    Duffet-Smith, P., and Zwart, J.,
    Practical Astronomy with your Calculator or Spreadsheet,
    4th edition, Cambridge University Press, 2011.

    Parameters
    ----------
    year : int
        Year as an integer. Years preceding 1 A.D. should be represented
        using astronomical year numbering, where year 0 is 1 B.C.

    month : int
        Month as an integer, where January is 1 and December is 12.

    day : float
        Day of the month. May contain a fractional component representing
        the time of day.

    Returns
    -------
    float
        Julian Day.

    Examples
    --------
    >>> date_to_jd(1985, 2, 17.25)
    2446113.75
    """

    if month <= 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # Julian calendar before the Gregorian reform.
    before_gregorian = (
        year < 1582
        or (year == 1582 and month < 10)
        or (year == 1582 and month == 10 and day < 15)
    )

    if before_gregorian:
        B = 0
    else:
        A = math.trunc(yearp / 100.0)
        B = 2 - A + math.trunc(A / 4.0)

    if yearp < 0:
        C = math.trunc(365.25 * yearp - 0.75)
    else:
        C = math.trunc(365.25 * yearp)

    D = math.trunc(30.6001 * (monthp + 1))

    return B + C + D + day + 1720994.5


def jd_to_date(jd):
    """
    Convert Julian Day to a calendar date.

    Algorithm from:

    Duffet-Smith, P., and Zwart, J.,
    Practical Astronomy with your Calculator or Spreadsheet,
    4th edition, Cambridge University Press, 2011.

    Parameters
    ----------
    jd : float
        Julian Day.

    Returns
    -------
    tuple
        Tuple containing year, month, and fractional day.

    Examples
    --------
    >>> jd_to_date(2446113.75)
    (1985, 2, 17.25)
    """

    jd += 0.5
    fractional, integer = math.modf(jd)
    integer = int(integer)

    A = math.trunc((integer - 1867216.25) / 36524.25)

    if integer > 2299160:
        B = integer + 1 + A - math.trunc(A / 4.0)
    else:
        B = integer

    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)

    day = C - E + fractional - math.trunc(30.6001 * G)

    month = G - 1 if G < 13.5 else G - 13
    year = D - 4716 if month > 2 else D - 4715

    return year, month, day


def jd_to_date2(jd):
    """
    Convert Julian Day to a formatted calendar date string.

    Parameters
    ----------
    jd : float
        Julian Day.

    Returns
    -------
    str
        Formatted date string.

    Examples
    --------
    >>> jd_to_date2(2446113.75)
    'Feb 17 1985  06:00:00.000000'
    """

    year, month, day = jd_to_date(jd)
    fractional_day, day_number = math.modf(day)

    hour, minute, second, microsecond = days_to_hmsm(
        fractional_day
    )

    months = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )

    return (
        f"{months[month - 1]} {int(day_number)} {year}  "
        f"{hour:02d}:{minute:02d}:{second:02d}.{microsecond:06d}"
    )


def hmsm_to_days(hour=0, mins=0, sec=0, micro=0):
    """
    Convert hours, minutes, seconds, and microseconds to fractional days.

    Parameters
    ----------
    hour : int, optional
        Hour number. Defaults to 0.

    mins : int, optional
        Minute number. Defaults to 0.

    sec : int, optional
        Second number. Defaults to 0.

    micro : int, optional
        Microsecond number. Defaults to 0.

    Returns
    -------
    float
        Fractional number of days.

    Examples
    --------
    >>> hmsm_to_days(hour=6)
    0.25
    """

    total_seconds = (
        hour * 3600
        + mins * 60
        + sec
        + micro / MICROSECONDS_PER_SECOND
    )

    return total_seconds / SECONDS_PER_DAY


def days_to_hmsm(days):
    """
    Convert fractional days to hours, minutes, seconds, and microseconds.

    Precision beyond microseconds is rounded to the nearest microsecond.

    Parameters
    ----------
    days : float
        Fractional days. Must satisfy 0 <= days < 1.

    Returns
    -------
    tuple
        Hour, minute, second, and microsecond.

    Raises
    ------
    ValueError
        If ``days`` is outside the interval [0, 1).

    Examples
    --------
    >>> days_to_hmsm(0.1)
    (2, 24, 0, 0)
    """

    if not 0.0 <= days < 1.0:
        raise ValueError("days must satisfy 0 <= days < 1")

    total_microseconds = round(days * MICROSECONDS_PER_DAY)

    hour, remainder = divmod(
        total_microseconds,
        MICROSECONDS_PER_HOUR,
    )

    minute, remainder = divmod(
        remainder,
        MICROSECONDS_PER_MINUTE,
    )

    second, microsecond = divmod(
        remainder,
        MICROSECONDS_PER_SECOND,
    )

    # A value extremely close to one day may round to midnight.
    if hour == 24:
        hour = minute = second = microsecond = 0

    return int(hour), int(minute), int(second), int(microsecond)


def datetime_to_jd(date):
    """
    Convert a datetime object to Julian Day.

    Parameters
    ----------
    date : datetime.datetime
        Datetime instance.

    Returns
    -------
    float
        Julian Day.

    Notes
    -----
    Naive datetime objects are assumed to represent the intended time scale.
    Timezone-aware datetime objects are not automatically converted to UTC.

    Examples
    --------
    >>> d = datetime(1985, 2, 17, 6)
    >>> datetime_to_jd(d)
    2446113.75
    """

    fractional_day = hmsm_to_days(
        date.hour,
        date.minute,
        date.second,
        date.microsecond,
    )

    return date_to_jd(
        date.year,
        date.month,
        date.day + fractional_day,
    )


def jd_to_datetime(jd):
    """
    Convert Julian Day to a datetime object.

    Parameters
    ----------
    jd : float
        Julian Day.

    Returns
    -------
    datetime
        Datetime equivalent of the Julian Day.

    Examples
    --------
    >>> jd_to_datetime(2446113.75)
    datetime(1985, 2, 17, 6, 0)
    """

    year, month, day = jd_to_date(jd)
    fractional_day, day_number = math.modf(day)

    total_microseconds = round(
        fractional_day * MICROSECONDS_PER_DAY
    )

    # Normalize rounding at the day boundary.
    if total_microseconds >= MICROSECONDS_PER_DAY:
        day_number += 1
        total_microseconds -= MICROSECONDS_PER_DAY
    elif total_microseconds < 0:
        day_number -= 1
        total_microseconds += MICROSECONDS_PER_DAY

    hour, remainder = divmod(
        total_microseconds,
        MICROSECONDS_PER_HOUR,
    )

    minute, remainder = divmod(
        remainder,
        MICROSECONDS_PER_MINUTE,
    )

    second, microsecond = divmod(
        remainder,
        MICROSECONDS_PER_SECOND,
    )

    return datetime(
        year,
        month,
        int(day_number),
        int(hour),
        int(minute),
        int(second),
        int(microsecond),
    )


def timedelta_to_days(td):
    """
    Convert a datetime.timedelta object to total days.

    Parameters
    ----------
    td : datetime.timedelta
        Timedelta instance.

    Returns
    -------
    float
        Total number of days represented by the timedelta.

    Examples
    --------
    >>> td = datetime.timedelta(days=4, seconds=43200)
    >>> timedelta_to_days(td)
    4.5
    """

    return td.total_seconds() / SECONDS_PER_DAY


class datetime(dt.datetime):
    """
    Datetime subclass supporting Julian Date conversions.

    Standard datetime arithmetic is used for addition and subtraction so that
    microsecond precision is preserved.

    Julian Date conversion is available through ``to_jd()`` and ``to_mjd()``.
    """

    def __add__(self, other):
        """Add a datetime.timedelta using native datetime arithmetic."""

        if not isinstance(other, dt.timedelta):
            return NotImplemented

        result = dt.datetime.__add__(self, other)

        return datetime(
            result.year,
            result.month,
            result.day,
            result.hour,
            result.minute,
            result.second,
            result.microsecond,
            tzinfo=result.tzinfo,
            fold=result.fold,
        )

    def __radd__(self, other):
        """Add a datetime.timedelta to this datetime."""
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract a timedelta or datetime.

        Timedelta subtraction returns this datetime subclass. Datetime
        subtraction returns a standard datetime.timedelta.
        """

        if isinstance(other, dt.timedelta):
            result = dt.datetime.__sub__(self, other)

            return datetime(
                result.year,
                result.month,
                result.day,
                result.hour,
                result.minute,
                result.second,
                result.microsecond,
                tzinfo=result.tzinfo,
                fold=result.fold,
            )

        if isinstance(other, dt.datetime):
            return dt.datetime.__sub__(self, other)

        return NotImplemented

    def __rsub__(self, other):
        """Subtract this datetime from another datetime."""

        if isinstance(other, dt.datetime):
            return dt.datetime.__sub__(other, self)

        return NotImplemented

    def to_jd(self):
        """Convert this datetime to Julian Day."""
        return datetime_to_jd(self)

    def to_mjd(self):
        """Convert this datetime to Modified Julian Day."""
        return jd_to_mjd(self.to_jd())


def JD_SPJ2000D(JD):
    """
    Convert Julian Day to seconds from the J2000 epoch.

    Parameters
    ----------
    JD : float
        Julian Day.

    Returns
    -------
    float
        Seconds from JD 2451545.0.
    """

    return (JD - JD_J2000) * SECONDS_PER_DAY


def Date_SPJ2000(day, month, year):
    """
    Convert a calendar date to seconds from the J2000 epoch.

    Parameters
    ----------
    day : float
        Day of the month, optionally including a fractional component.

    month : int
        Calendar month.

    year : int
        Calendar year.

    Returns
    -------
    float
        Seconds from the J2000 epoch.
    """

    return JD_SPJ2000D(
        date_to_jd(year, month, day)
    )