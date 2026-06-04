try:
    from prettytable import PrettyTable
except ModuleNotFoundError:
    class PrettyTable(object):
        def __init__(self, field_names):
            self.field_names = list(field_names)
            self.rows = []
            self.custom_format = {}

        def add_row(self, row):
            self.rows.append(list(row))

        def _format_cell(self, field, value):
            formatter = self.custom_format.get(field)
            if formatter is not None:
                return formatter(field, value)
            return str(value)

        def __str__(self):
            formatted = [[str(field) for field in self.field_names]]
            for row in self.rows:
                formatted.append(
                    [
                        str(cell) if idx >= len(self.field_names) else self._format_cell(self.field_names[idx], cell)
                        for idx, cell in enumerate(row)
                    ]
                )
            widths = [max(len(row[idx]) for row in formatted) for idx in range(len(self.field_names))]
            lines = []
            for row in formatted:
                lines.append(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(widths))))
            return "\n".join(lines)
