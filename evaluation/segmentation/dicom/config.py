# config.py
"""Pydicom configuration options."""
# Copyright (c) 2012 Darcy Mason
# This file is part of pydicom, released under a modified MIT license.
#    See the file license.txt included with this distribution, also
#    available at http://pydicom.googlecode.com

# doc strings following items are picked up by sphinx for documentation

# Set the type used to hold DS values

import logging

# Set the type used to hold DS values
#    default False; was decimal-based in pydicom 0.9.7
use_DS_decimal = False


data_element_callback = None
"""Set data_element_callback to a function to be called from read_dataset
every time a RawDataElement has been returned, before it is added
to the dataset.
"""

data_element_callback_kwargs = {}
"""Set this to use as keyword arguments passed to the data_element_callback
function"""


def reset_data_element_callback():
    global data_element_callback
    global data_element_callback_kwargs
    data_element_callback = None
    data_element_callback_kwargs = {}


def DS_decimal(use_Decimal_boolean=True):
    """Set DS class to be derived from Decimal (True) or from float (False)
    If this function is never called, the default in pydicom >= 0.9.8
    is for DS to be based on float.
    """
    use_DS_decimal = use_Decimal_boolean
    import dicom.valuerep as valuerep
    if use_DS_decimal:
        valuerep.DSclass = valuerep.DSdecimal
    else:
        valuerep.DSclass = valuerep.DSfloat


# Configuration flags
allow_DS_float = False
"""Set allow_float to True to allow DSdecimal instances
to be created with floats; otherwise, they must be explicitly
converted to strings, with the user explicity setting the
precision of digits and rounding. Default: False"""

enforce_valid_values = False
"""Raise errors if any value is not allowed by DICOM standard,
e.g. DS strings that are longer than 16 characters;
IS strings outside the allowed range.
"""

datetime_conversion = False
"""Set datetime_conversion to convert DA, DT and TM
data elements to datetime.date, datetime.datetime
and datetime.time respectively. Default: False
"""

# Logging system and debug function to change logging level
logger = logging.getLogger('pydicom')
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


image_handlers = []
"""Image handlers for converting pixel data.
This is an ordered list that the dataset._get_pixel_array()
method will try to extract a correctly sized numpy array from the
PixelData attribute.
If a handler lacks required dependencies or can not otherwise be loaded,
it shall throw an ImportError.
Handers shall have two methods:

supports_transfer_syntax(dicom_dataset)
  This returns True if the handler might support the transfer syntax
  indicated in the dicom_dataset

def get_pixeldata(dicom_dataset):
  This shall either throw an exception or return a correctly sized numpy
  array derived from the PixelData.  Reshaping the array to the correct
  dimensions is handled outside the image handler

The first handler that both announces that it supports the transfer syntax
and does not throw an exception, either in getting the data or when the data
is reshaped to the correct dimensions, is the handler that will provide the
data.

If they all fail, the last one to throw an exception gets to see its
exception thrown up.

If no one throws an exception, but they all refuse to support the transfer
syntax, then this fact is announced in a NotImplementedError exception.
"""

have_numpy = True
try:
    import dicom.pixel_data_handlers.numpy_handler as numpy_handler
    image_handlers.append(numpy_handler)

    import dicom.pixel_data_handlers.rle_handler as rle_handler
    image_handlers.append(rle_handler)

except ImportError as e:
    logger.debug("Could not import numpy")
    have_numpy = False

have_pillow = True
try:
    import dicom.pixel_data_handlers.pillow_handler as pillow_handler
    image_handlers.append(pillow_handler)
except ImportError as e:
    logger.debug("Could not import pillow")
    have_pillow = False

have_jpeg_ls = True
try:
    import dicom.pixel_data_handlers.jpeg_ls_handler as jpeg_ls_handler
    image_handlers.append(jpeg_ls_handler)
except ImportError as e:
    logger.debug("Could not import jpeg_ls")
    have_jpeg_ls = False

have_gdcm = True
try:
    import pixel_data_handlers.gdcm_handler as gdcm_handler
    image_handlers.append(gdcm_handler)
except ImportError as e:
    logger.debug("Could not import gdcm")
    have_gdcm = False


def debug(debug_on=True):
    """Turn debugging of DICOM file reading and writing on or off.
    When debugging is on, file location and details about the
    elements read at that location are logged to the 'pydicom'
    logger using python's logging module.

    :param debug_on: True (default) to turn on debugging,
    False to turn off.
    """
    global logger, debugging
    if debug_on:
        logger.setLevel(logging.DEBUG)
        debugging = True
    else:
        logger.setLevel(logging.WARNING)
        debugging = False


# force level=WARNING, in case logging default is set differently (issue 103)
debug(False)
