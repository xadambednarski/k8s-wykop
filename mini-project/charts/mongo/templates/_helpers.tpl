{{- define "mongo.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "mongo.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "mongo.name" .) -}}
{{- end -}}