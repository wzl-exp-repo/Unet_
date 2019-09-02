'''
Danijar Hafner

 @misc{hafner2019reuselayers,
  author = {Hafner, Danijar},
  title = {Structuring Deep Learning Models},
  year = {2019},
  howpublished = {Blog post},
  url = {https://danijar.com/structuring-models/}
}
'''
class ReuseElements(object):

  def __init__(self, elements):
    self._elements = elements
    self._adding = (len(elements) == 0)
    self._index = 0

  def __call__(self, provided):
    if self._adding:
      self._elements.append(provided)
      return provided
    existing = self._elements[self._index]
    self._index += 1
    assert isinstance(existing, type(provided))
    return existing