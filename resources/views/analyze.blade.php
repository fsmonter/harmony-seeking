<form action="/analyze" method="POST" enctype="multipart/form-data">
    @csrf
    <input type="file" name="image" required>
    <button type="submit">Analyze Image</button>
  </form>
